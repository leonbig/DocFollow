from typing import List, Dict, Tuple, Optional, Any
import datetime
from openai import OpenAI
from tools.functions import retry_on_failure
from tools.splitting import spacy_sentence_split  
from tools.mapping import direct_sentence_matching_one2one  
from tools.predefined_category import RELATION_DEFINITIONS, LOGICAL_RELATIONS

class SentenceRelaEvaluator:
    def __init__(self, client: OpenAI, eval_model: str, log_file: Optional[str] = None):

        if not isinstance(client, OpenAI):
            raise TypeError("client must be an instance of OpenAI")
        if not isinstance(eval_model, str) or not eval_model.strip():
            raise ValueError("eval_model must be a non-empty string")
            
        self.client = client
        self.eval_model = eval_model
        self.log_file = log_file

    def log_message(self, message: str, log_file: str = None):
        """Log a message with timestamp to console and optionally to a file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')
 
    @retry_on_failure(max_retries=3)
    def split_sentences(self, orig_text: str, para_text: str) -> tuple[list[str], list[str]]:
        """使用spaCy分割句子"""
        try:
            # 使用现有的spacy_sentence_split函数处理两段文本
            orig_sentences = spacy_sentence_split(orig_text)
            para_sentences = spacy_sentence_split(para_text)
            return orig_sentences, para_sentences
            
        except Exception as e:
            print(f"Error in sentence splitting: {e}")
            return [], []
    
    @retry_on_failure(max_retries=3)
    def direct_sentence_matching(self, source: List[str], target: List[str]) -> Dict[int, int]:
        return direct_sentence_matching_one2one(source, target)

    @retry_on_failure(max_retries=3)
    def extract_relations(self, sentences: List[str]) -> List[Tuple[int, int, str]]:
        relations = []
        for i in range(len(sentences) - 1):
            rel = self._get_relation(sentences[i], sentences[i+1])
            relations.append((i, i+1, rel))
        return relations
    
    @retry_on_failure(max_retries=3)
    def _get_relation(self, sent1: str, sent2: str) -> str:
        
        prompt = f"""You are an expert linguist specialized in discourse analysis. Your task is to identify the primary relationship between two consecutive sentences based on a predefined list of functions. Follow these instructions precisely:

        1.  **Analyze Function:** Determine the main communicative function of Sentence 2 in relation to Sentence 1.
        2.  **Use Definitions:** Base your classification *strictly* on the relationship definitions provided below.
        3.  **Select Exactly One:** Choose the single best relationship type from the list that accurately describes this primary function.
        4.  **Verbatim Response:** Your response must consist ONLY of the chosen relationship name, exactly as it appears in the list. Do not include explanations, justifications, or any other text.
        5.  **"Others" Category:** Only select the "Others" category if none of the other relationship types reasonably apply. Exhaust all other possibilities before choosing "Others".

        **Available Relationship Types and Definitions:**
        {RELATION_DEFINITIONS}
        **Classification Task:**

        Carefully analyze the relationship between Sentence 1 and Sentence 2. Choose ONLY ONE relationship from the following list:{", ".join(LOGICAL_RELATIONS)}

        **Selection Process:**
        - Evaluate Sentence 2's primary function relative to Sentence 1 against the provided definitions.
        - Try to match with one of the specific relationship types (Causality, Condition, Purpose, etc.) first.
        - Select "Others" ONLY if all other relationship types have been considered and none of them fit the relationship.
        - Always prefer a more specific category over the "Others" category when possible.
      
        **Input Sentences:**
        Sentence 1: {sent1}
        Sentence 2: {sent2}

        **Relationship:**"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except:
            return "Unknown"

    @retry_on_failure(max_retries=3)
    def evaluate_pair(self, original: str, paraphrase: str) -> tuple[float, list[str], list[str], dict[int, int]]:

        self.log_message("\n" + "="*50, self.log_file)
        self.log_message("Sentence Logic Evaluation Details:", self.log_file)
        self.log_message("-"*50, self.log_file)
        
        # Log input text information
        self.log_message(f"Original text preview: {original[:100]}...", self.log_file)
        self.log_message(f"Paraphrased text preview: {paraphrase[:100]}...", self.log_file)
        
        try:
            # Preprocessing
            self.log_message("Splitting sentences...", self.log_file)
            orig_sents, para_sents = self.split_sentences(original, paraphrase)
            self.log_message(f"Split into {len(orig_sents)} original and {len(para_sents)} paraphrased sentences", self.log_file)
            
            # Log sentence splitting results
            self.log_message("\nOriginal sentences:", self.log_file)
            for i, sent in enumerate(orig_sents):
                self.log_message(f"[{i}] {sent}", self.log_file)
            self.log_message("\nParaphrased sentences:", self.log_file)
            for i, sent in enumerate(para_sents):
                self.log_message(f"[{i}] {sent}", self.log_file)

            # Sentence alignment
            self.log_message("\nPerforming sentence alignment...", self.log_file)
            alignment = self.direct_sentence_matching(orig_sents, para_sents)
            self.log_message(f"Found {len(alignment)} sentence alignments", self.log_file)
            
            # Log alignment results
            self.log_message("\nAlignment results:", self.log_file)
            for para_idx, orig_idx in alignment.items():
                if orig_idx is not None:
                    self.log_message(f"Original[{orig_idx}] -> Paraphrase[{para_idx}]", self.log_file)
                    self.log_message(f"  Original: {orig_sents[orig_idx]}", self.log_file)
                    self.log_message(f"  Paraphrase: {para_sents[para_idx]}", self.log_file)

            # Extract relationships
            self.log_message("\nExtracting sentence relationships...", self.log_file)
            orig_relations = self.extract_relations(orig_sents)
            para_relations = self.extract_relations(para_sents)
            self.log_message(f"Extracted {len(orig_relations)} original and {len(para_relations)} paraphrased relations", self.log_file)
            
            # Log relation extraction results
            self.log_message("\nOriginal text relations:", self.log_file)
            for o1, o2, rel in orig_relations:
                self.log_message(f"Sentences [{o1}]-[{o2}]: {rel}", self.log_file)
                self.log_message(f"  [{o1}] {orig_sents[o1]}", self.log_file)
                self.log_message(f"  [{o2}] {orig_sents[o2]}", self.log_file)
                
            self.log_message("\nParaphrased text relations:", self.log_file)
            for p1, p2, rel in para_relations:
                self.log_message(f"Sentences [{p1}]-[{p2}]: {rel}", self.log_file)
                self.log_message(f"  [{p1}] {para_sents[p1]}", self.log_file)
                self.log_message(f"  [{p2}] {para_sents[p2]}", self.log_file)

            # Cross-text comparison
            self.log_message("\nComparing relations...", self.log_file)
            matched = 0


            for p1, p2, rel in para_relations:
                o1 = alignment.get(p1)
                o2 = alignment.get(p2)

                if o1 is not None and o2 is not None:
                    orig_rel = next((r for s1, s2, r in orig_relations if s1 == o1 and s2 == o2), None)
                    if orig_rel:
                        if rel == orig_rel:
                            matched += 1
                            self.log_message(f"Matched relation between sentences {p1}-{p2}: {rel}", self.log_file)
                        else:
                            self.log_message(f"Mismatched relation between sentences {p1}-{p2}: {rel} vs {orig_rel}", self.log_file)
                        
            # Calculate preservation score with improved error handling
            try:
                if len(para_relations) <= 0:
                    self.log_message("No paraphrased relations found, score=0.0", self.log_file)
                    return 0.0, orig_sents, para_sents, alignment
                
                if matched < 0:
                    self.log_message("Warning: Negative match count detected, setting to 0", self.log_file)
                    matched = 0
                
                score = matched / len(para_relations)
                
                self.log_message(f"Final sentence relation score: {score:.2f}", self.log_file)
                self.log_message("="*50 + "\n", self.log_file)
                return score, orig_sents, para_sents, alignment
                
            except Exception as e:
                self.log_message(f"Error calculating final score: {str(e)}", self.log_file)
                self.log_message("="*50 + "\n", self.log_file)
                return 0.0, orig_sents, para_sents, alignment
            
        except Exception as e:
            self.log_message(f"Error: Evaluation failed - {str(e)}", self.log_file)
            self.log_message("="*50 + "\n", self.log_file)
            return 0.0, [], [], {}
