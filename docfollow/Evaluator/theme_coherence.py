from typing import List, Dict, Any, Optional
import json_repair
from openai import OpenAI
import datetime
import spacy
from langdetect import detect, LangDetectException
import concurrent.futures
from tools.functions import retry_on_failure

class ThemeCoherenceEvaluator:
    """Theme coherence evaluator using an LLM provider to evaluate text"""

    def __init__(self, client: OpenAI, eval_model: str, log_file: str):
        """Initialize the text analyzer with an LLM provider"""
        self.client = client
        self.eval_model = eval_model
        self.log_file = log_file
        self.loaded_models: Dict[str, spacy.Language] = {}

    def log_message(self, message: str, log_file: str = None):
        """记录日志消息，使用线程锁确保线程安全"""
        import threading
        
        # 使用类变量作为锁，确保只创建一次
        if not hasattr(self, '_log_lock'):
            self._log_lock = threading.Lock()
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        # 使用锁确保线程安全的文件写入
        with self._log_lock:
            if log_file:
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(log_msg + '\n')
                except Exception as e:
                    print(f"Error writing to log file: {e}")

    @retry_on_failure(max_retries=3)
    def _get_nlp(self, language: str) -> Optional[spacy.Language]:
        """Loads and returns the spaCy model for the specified language, caching it."""
        supported_languages = ['en', 'zh']
        if language not in supported_languages:
            self.log_message(f"Detected language '{language}' is not directly supported for spaCy model loading (supported: {supported_languages}).", self.log_file)
            self.log_message("Defaulting to English ('en') model.", self.log_file)
            language = 'en'

        if language in self.loaded_models:
            return self.loaded_models[language]
        else:
            model_name = f"{language}_core_web_sm"
            try:
                self.log_message(f"Attempting to load spaCy model '{model_name}'...", self.log_file)
                nlp = spacy.load(model_name)
                self.loaded_models[language] = nlp
                self.log_message(f"SpaCy model '{model_name}' loaded successfully.", self.log_file)
                return nlp
            except OSError as e:
                self.log_message(f"Error: spaCy model '{model_name}' not found: {e}", self.log_file)
                return None
            except Exception as e:
                self.log_message(f"An unexpected error occurred while loading spaCy model '{model_name}': {e}", self.log_file)
                return None

    @retry_on_failure(max_retries=3)
    def evaluate_sentence(self, theme: str, sentence: str, context: str) -> Dict[str, Any]:
        """Evaluate a sentence using OpenAI with retry logic."""
        prompt = f"""
        Please evaluate this sentence based on the following three criteria: relevance, appropriateness, and coherence with context.

        Theme: {theme}
        Sentence: {sentence}
        Surrounding Context: {context}

        For each criterion, determine if the sentence meets the standard (Yes/No), provide a brief explanation, and list any inappropriate words or phrases that cause issues.

        Relevance: Is the sentence directly related to the `Theme` or intended subtopic?
        Appropriateness: Is the sentence's language, style, and tone suitable for the given `Theme` and consistent with the `Surrounding Context`? Does it avoid jarring shifts in formality, vocabulary, or tone compared to the surrounding text?
        Coherence: Does the sentence logically follow from or connect to the `Surrounding Context` while maintaining consistency with the `Theme`?

        Respond in JSON format:
        {{
            "relevance": {{
                "assessment": "Yes/No",
                "explanation": "Brief reason",
                "inappropriate_words": ["word1", "word2", ...]
            }},
            "appropriateness": {{
                "assessment": "Yes/No",
                "explanation": "Brief reason",
                "inappropriate_words": ["word1", "word2", ...]
            }},
            "coherence": {{
                "assessment": "Yes/No",
                "explanation": "Brief reason",
                "inappropriate_words": ["word1", "word2", ...]
            }}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {"role": "system", "content": "You are an academic text analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json_repair.loads(response.choices[0].message.content.strip())
            return result
        except Exception as e:
            self.log_message(f"Error during sentence evaluation: {e}", self.log_file)
            raise
    @retry_on_failure(max_retries=3)
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        try:
            # 使用已加载的模型或加载新模型
            nlp = self._get_nlp('en')
            if not nlp:
                return [s.strip() for s in text.split('\n') if s.strip()]
            
            # 处理文本并提取句子
            doc = nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
        except Exception as e:
            self.log_message(f"Sentence splitting failed: {e}. Using fallback.", self.log_file)
            return [s.strip() for s in text.split('\n') if s.strip()]

    def get_context(self, sentences: List[str], sentence: str, window_size: int = 2) -> str:
        """Get the surrounding context of a sentence"""
        try:
            index = sentences.index(sentence)
        except ValueError:
            self.log_message(f"Warning: Sentence not found exactly in LLM split results: '{sentence[:50]}...'", self.log_file)
            return "Context unavailable (sentence not matched exactly)"

        start = max(0, index - window_size)
        end = min(len(sentences), index + window_size + 1)

        context_sentences = sentences[start:index] + sentences[index+1:end]
        return " ".join(context_sentences)

    def process_sentence(self, theme: str, sentence: str, sentences: List[str], i: int, nlp_model, target_language: str):
        """处理单个句子的评估，用于并行执行"""
        if not sentence.strip():
            return None
            
        self.log_message(f"\nProcessing sentence {i+1}/{len(sentences)}: '{sentence[:80]}...'", self.log_file)
        context = self.get_context(sentences, sentence)
        
        try:
            evaluation = self.evaluate_sentence(theme, sentence, context)
            
            if evaluation.get("relevance", {}).get("assessment") == "Error":
                self.log_message(f"Skipping score calculation for sentence {i+1} due to evaluation error.", self.log_file)
                return {
                    "sentence": sentence,
                    "inappropriateWords": [],
                    "evaluation": evaluation,
                    "sentence_score": 0.0
                }

            inappropriate_words = set()
            for criterion in ["relevance", "appropriateness", "coherence"]:
                inappropriate_words.update(evaluation.get(criterion, {}).get("inappropriate_words", []))

            try:
                sentence_doc = nlp_model(sentence)
                num_sentence_words = len([token for token in sentence_doc if not token.is_punct and not token.is_space])

                inappropriate_word_count_in_sentence = 0
                if num_sentence_words > 0:
                    for item in inappropriate_words:
                        if item.lower() in sentence.lower():
                            occurrences = sentence.lower().count(item.lower())
                            words_in_item = len(item.lower().split())
                            inappropriate_word_count_in_sentence += occurrences * words_in_item

                    sentence_score = max(0.0, inappropriate_word_count_in_sentence / num_sentence_words)
                else:
                    sentence_score = 0.0

                self.log_message(f"Sentence word count (via {target_language.upper()} tokenizer): {num_sentence_words}", self.log_file)
                self.log_message(f"Sentence score (inappropriate/total words): {sentence_score:.4f}", self.log_file)

            except Exception as e:
                self.log_message(f"Error calculating score for sentence '{sentence[:50]}...': {e}", self.log_file)
                sentence_score = 0.0

            return {
                "sentence": sentence,
                "inappropriateWords": list(inappropriate_words),
                "evaluation": evaluation,
                "sentence_score": sentence_score
            }

        except Exception as e:
            self.log_message(f"Error processing sentence {i+1}: {e}", self.log_file)
            return None

    def evaluate_text(self, theme: str, text: str, max_workers: int = 5):
        """Evaluate text against a title for relevance, identify inappropriate words,
           and calculate scores based on inappropriate word proportion."""
        self.log_message("\n" + "="*50, self.log_file)
        self.log_message("Topic Coherence Evaluation Details", self.log_file)
        self.log_message("-"*50, self.log_file)
        self.log_message(f"Evaluating text with theme: '{theme}'", self.log_file)
        self.log_message(f"Text preview: {text[:100]}...", self.log_file)
        self.log_message(f"Text length: {len(text)} chars", self.log_file)

        try:
            text_sample = text[:500]
            if text_sample.strip():
                detected_language_code = detect(text_sample)
                self.log_message(f"Detected language code: '{detected_language_code}'", self.log_file)
            else:
                self.log_message("Text sample is empty, cannot detect language. Defaulting to 'en'.", self.log_file)
                detected_language_code = 'en'
        except LangDetectException:
            self.log_message("Language detection failed (text might be too short or ambiguous). Defaulting to 'en'.", self.log_file)
            detected_language_code = 'en'
        except Exception as e:
            self.log_message(f"An unexpected error occurred during language detection: {e}. Aborting.", self.log_file)
            return None

        target_language = None
        if detected_language_code:
            if detected_language_code.startswith('en'):
                target_language = 'en'
            elif detected_language_code.startswith('zh'):
                target_language = 'zh'
            else:
                self.log_message(f"Detected language '{detected_language_code}' not mapped to 'en' or 'zh'. Defaulting to 'en'.", self.log_file)
                target_language = 'en'

        if not target_language:
            self.log_message("Error: Could not determine target language. Aborting.", self.log_file)
            return None

        self.log_message(f"Using spaCy model for language: '{target_language.upper()}'", self.log_file)

        nlp_model = self._get_nlp(target_language)
        if nlp_model is None:
            self.log_message(f"Failed to load necessary spaCy model for '{target_language}'. Aborting evaluation.", self.log_file)
            return None

        sentences = self.split_into_sentences(text)
        if not sentences:
            self.log_message("No valid sentences found in text. Aborting evaluation.", self.log_file)
            return None

        results = []
        total_score = 0.0
        self.log_message(f"Processing {len(sentences)} sentences with {max_workers} parallel workers...", self.log_file)

        # 使用线程池并行处理句子
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sentence = {
                executor.submit(self.process_sentence, theme, sentence, sentences, i, nlp_model, target_language): (i, sentence)
                for i, sentence in enumerate(sentences) if sentence.strip()
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_sentence):
                i, sentence = future_to_sentence[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        total_score += result["sentence_score"]
                except Exception as e:
                    self.log_message(f"Exception processing sentence {i+1}: {e}", self.log_file)

        if not results:
            self.log_message("No valid results obtained. Returning 0.0 as score.", self.log_file)
            return 0.0

        average_score = total_score / len(results)
        
        final_result = {
            "sentence_results": results,
            "average_score": average_score
        }

        self.log_message(f"\nTotal score accumulated: {total_score:.4f}", self.log_file)
        self.log_message(f"Final theme coherence score: {final_result['average_score']:.4f}", self.log_file)
        self.log_message("="*50 + "\n", self.log_file)

        return average_score