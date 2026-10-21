from typing import List, Dict, Tuple, Optional, Any
import json_repair
import datetime
from tools.mapping import direct_sentence_matching_llm
from tools.mapping import entity_match_scoring, classify_text_domain
from tools.functions import retry_on_failure
from tools.predefined_category import get_entity_types_by_domain

class EntityRelationStore:
    def __init__(self):
        self.entities = {} 

class EntityEvaluator:
    def __init__(self, client, eval_model, log_file):
        self.client = client
        self.eval_model = eval_model
        self.log_file = log_file
        self.entity_relation_store = EntityRelationStore()
        self.domain = None  
    
    def log_message(self, message: str, log_file: str = None):
        """Log a message with timestamp to console and optionally to a file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')

    @retry_on_failure(max_retries=3)
    def direct_sentence_mapping(self, source: List[str], target: List[str]) -> Dict[int, int]:
        return direct_sentence_matching_llm(self.client, self.eval_model, source, target)

    @retry_on_failure(max_retries=3)
    def classify_text_domain(self, text: str) -> str:
        return classify_text_domain(self.client, self.eval_model, text, self.log_message, self.log_file)
            

    @retry_on_failure(max_retries=3)
    def extract_entities(self, sent1: str, sent2: str, orig_idx: int, para_idx: int) -> None: 
        text_domain = self.domain
        
        domain_entity_definitions = get_entity_types_by_domain(text_domain)
        
        formatted_entity_definitions = "\n".join([f"- {item['category']}: {item['definition']}" for item in domain_entity_definitions])
        
        self.log_message(f"Using domain-specific entity types for domain: {text_domain}", self.log_file)

        entity_prompt = f"""
            Objective: Extract and classify entities from TWO input sentences with equal thoroughness. You MUST analyze BOTH sentences and output their entities in a single response.

            Entity Taxonomy and Definitions for domain '{text_domain}':
            {formatted_entity_definitions}

            === SENTENCE PAIR TO ANALYZE ===
            First Sentence: "{sent1}"
            Second Sentence: "{sent2}"

            Processing Instructions:
            1.  **Independent Analysis:** Process BOTH sentences separately but with equal thoroughness. You MUST analyze BOTH sentences and provide entities for each.
            2.  **Entity Identification:** For each sentence:
                - Identify all potential entities
                - Consider every word/phrase that could be an entity
                - Do not skip or ignore any potential entities
            3.  **Classification Rules:**
                - Use ONLY the entity types defined in the taxonomy above
                - Match each entity to the most specific applicable category
                - Consider ALL possible categories before making a decision
            4.  **OTHERS Category Usage:** Use ONLY when:
                - The term is clearly an entity
                - You have exhaustively considered all other categories
                - No other category definitively fits
            5.  **Completeness Check:** Before finalizing output, verify:
                - Both sentences have been fully analyzed
                - All potential entities have been considered
                - Each identified entity has been properly classified
                - Each sentence's entities have been analyzed with equal depth

            Output Structure:
            {{
                "result": [
                    {{
                        "sentence_index": 0,
                        "original_sentence": "First sentence text",
                        "entities": [
                            {{
                                "text": "exact_entity_text",
                                "type": "ENTITY_TYPE"
                            }}
                        ]
                    }},
                    {{
                        "sentence_index": 1,
                        "original_sentence": "Second sentence text",
                        "entities": [
                            {{
                                "text": "exact_entity_text",
                                "type": "ENTITY_TYPE"
                            }}
                        ]
                    }}
                ]
            }}

            FINAL CHECK:
            - Have you analyzed BOTH sentences?
            - Does your output contain EXACTLY TWO objects in the result array?
            - Are all entity texts exact matches from the sentences?
            - Have you considered all possible entities in both sentences?
            - Have you applied the same analysis depth to both sentences?
            - Is your output wrapped in a JSON object with a "result" field containing the array?

            Key Notes:
            - Apply the SAME taxonomy and classification logic rigorously to BOTH sentences
            - Adhere strictly to the provided entity definitions
            - Prioritize using specific entity categories first
            - Use 'OTHERS' only when no other category is appropriate
            - Ensure valid JSON syntax
        """

        try:
            entity_resp = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a precise entity extraction system that ALWAYS analyzes BOTH input sentences.
Your task is to extract entities from TWO input sentences. You must:
1. Process BOTH sentences independently
2. Extract ALL possible entities from EACH sentence
3. Return a JSON object with a "result" field containing an array of EXACTLY TWO objects
4. Never skip or ignore either sentence
5. Apply the same level of analysis to both sentences
6. Ensure each output object has the correct sentence_index (0 for first, 1 for second)
7. Follow the entity taxonomy strictly as provided
Remember: Your output must be a JSON object with format: {"result": [sentence1_obj, sentence2_obj]}"""
                    },
                    {
                        "role": "user", 
                        "content": entity_prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_content = entity_resp.choices[0].message.content.strip()
            self.log_message(f"raw_content: {raw_content}", self.log_file)
            parsed_data = json_repair.loads(raw_content)
            self.log_message(f"parsed_data: {parsed_data}", self.log_file)
            # Extract entities from the parsed data
            orig_entities = []
            para_entities = []
            
            if isinstance(parsed_data, dict) and "result" in parsed_data:
                # If output is a JSON object with a "result" key containing a list
                result_list = parsed_data["result"]
                if len(result_list) >= 2:
                    orig_entities = result_list[0].get("entities", [])
                    para_entities = result_list[1].get("entities", [])
            elif isinstance(parsed_data, list):
                if len(parsed_data) >= 2:
                    # Output is already a list with at least 2 items
                    orig_entities = parsed_data[0].get("entities", [])
                    para_entities = parsed_data[1].get("entities", [])
            elif isinstance(parsed_data, dict) and len(parsed_data) >= 2:
                # If output consists of two separate JSON objects
                # Convert to a list and process in order
                result_list = list(parsed_data.values())
                orig_entities = result_list[0].get("entities", []) if isinstance(result_list[0], dict) else []
                para_entities = result_list[1].get("entities", []) if isinstance(result_list[1], dict) and len(result_list) > 1 else []
            else:
                print(f"Warning: Unexpected structure in parsed entity data for indices {orig_idx}, {para_idx}: {parsed_data}")
            
            self.log_message(f"orig_entities: {orig_entities}", self.log_file)
            self.log_message(f"para_entities: {para_entities}", self.log_file)
            # Store the extracted entities
            self.entity_relation_store.entities.setdefault('orig', []).append({orig_idx: orig_entities})
            self.entity_relation_store.entities.setdefault('para', []).append({para_idx: para_entities})

        except Exception as e:
            print(f"Error processing entity extraction: {str(e)}")
            self.entity_relation_store.entities.setdefault('orig', []).append({orig_idx: []})
            self.entity_relation_store.entities.setdefault('para', []).append({para_idx: []})


    @retry_on_failure(max_retries=3)
    def _entity_match_score(self, orig_sent, para_sent, orig_entities: List[Dict], para_entities: List[Dict]) -> float:
        # Call the entity_match_scoring function from mapping module
        return entity_match_scoring(
            self.client, 
            self.eval_model,
            orig_sent,
            para_sent,
            orig_entities,
            para_entities,
            log_message=self.log_message,
            log_file=self.log_file
        )

    @retry_on_failure(max_retries=3)
    def evaluate_pair(self, orig_sents: List[str], para_sents: List[str]) -> Tuple[float, Dict[int, int], Dict[int, Dict[str, str]], Dict[int, Dict[str, List[Dict]]], str]:

        self.log_message("\n" + "="*50, self.log_file)
        self.log_message("Entity Evaluation Details:", self.log_file)
        self.log_message("-"*50, self.log_file)
        
        # 首先将原始文本和改写文本合并，用于领域分类
        orig_text = " ".join(orig_sents)
        para_text = " ".join(para_sents)
        combined_text = f"{orig_text} {para_text}"
        
        # 对整个文本进行领域分类，而不是针对单个句子
        self.log_message("\nClassifying text domain...", self.log_file)
        self.domain = self.classify_text_domain(combined_text)
        self.log_message(f"Text classified as domain: {self.domain}", self.log_file)
        
        # Split texts into sentences
        self.log_message("\nSplitting sentences...", self.log_file)
        try:
            self.log_message(f"\nSplit into {len(orig_sents)} original and {len(para_sents)} paraphrased sentences", self.log_file)
            self.log_message("\nOriginal sentences:", self.log_file)
            for i, sent in enumerate(orig_sents):
                self.log_message(f"[{i}] {sent}", self.log_file)

            self.log_message("\nParaphrased sentences:", self.log_file)
            for i, sent in enumerate(para_sents):
                self.log_message(f"[{i}] {sent}", self.log_file)
        except Exception as e:
            self.log_message(f"Error during sentence splitting: {str(e)}", self.log_file)
            return 0.0, {}, {}, {}
  
        # 使用LLM进行句子对齐
        self.log_message("\nPerforming sentence alignment using LLM...", self.log_file)
        try:
            alignment = self.direct_sentence_mapping(orig_sents, para_sents)
            self.log_message("\nAlignment results:", self.log_file)
            self.log_message(f"\nFound {len(alignment)} sentence alignments", self.log_file)
            for para_idx, orig_idx in alignment.items():
                self.log_message(f"Original[{orig_idx}] -> Paraphrase[{para_idx}]", self.log_file)
                self.log_message(f"Paraphrase: {para_sents[para_idx]}", self.log_file)
                if orig_idx is not None:
                    self.log_message(f"Original: {orig_sents[orig_idx]}", self.log_file)

        except Exception as e:
            self.log_message(f"Error during sentence alignment: {str(e)}", self.log_file)
            return 0.0, {}, {}, {}

        # Scoring initialization
        total_score = 0.0
        entities = {}
        sents = {}
        
        self.log_message("\nProcessing sentence pairs...", self.log_file)
        # Process each aligned sentence pair
        orig_other_idx = 0
        para_other_idx = 0
        for para_idx, orig_idx in alignment.items():
            try:
                orig_sent, para_sent = orig_sents[orig_idx], para_sents[para_idx]
                self.log_message(f"\nProcessing sentence pair [{orig_idx}]-[{para_idx}]:", self.log_file)
                self.log_message(f"Original: {orig_sent}", self.log_file)
                self.log_message(f"Paraphrase: {para_sent}", self.log_file)
                
                sents[para_idx] = {"original": orig_sent, "paraphrase": para_sent}

                # Extract entities and relations from both sentences
                self.log_message("Extracting entities...", self.log_file)
                self.extract_entities(orig_sent, para_sent, orig_idx, para_idx)          
                # Retrieve stored entities/relations
                orig_ents = self.entity_relation_store.entities['orig'][orig_other_idx].get(orig_idx, [])
                orig_other_idx += 1
                para_ents = self.entity_relation_store.entities['para'][para_other_idx].get(para_idx, [])
                para_other_idx += 1

                # 记录实体提取结果
                self.log_message("\nExtracted entities:", self.log_file)
                self.log_message("Original entities:", self.log_file)
                for ent in orig_ents:
                    self.log_message(f"  - {ent['text']} ({ent['type']})", self.log_file)
                self.log_message("Paraphrased entities:", self.log_file)
                for ent in para_ents:
                    self.log_message(f"  - {ent['text']} ({ent['type']})", self.log_file)
                                
                entities[para_idx] = {"original": orig_ents, "paraphrase": para_ents}

                # Calculate entity matching score
                self.log_message("\nCalculating entity matching score...", self.log_file)
                ent_score = self._entity_match_score(orig_sent, para_sent, orig_ents, para_ents)
                self.log_message(f"Entity matching score: {ent_score:.2f}", self.log_file)
                
                # Combined weighted score
                sentence_score = ent_score
                total_score += sentence_score
                self.log_message(f"Sentence pair score: {sentence_score:.2f}", self.log_file)

            except Exception as e:
                self.log_message(f"Error processing sentence pair: {str(e)}", self.log_file)
                continue

        # Calculate final average score
        final_score = total_score / len(alignment) if alignment else 0.0
        self.log_message(f"\nFinal entity evaluation score: {final_score:.2f}", self.log_file)
        self.log_message("="*50 + "\n", self.log_file)

        return final_score, alignment, sents, entities, self.domain
