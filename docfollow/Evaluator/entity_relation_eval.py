import json
from typing import List, Dict
import json_repair
from textwrap import dedent
import datetime
from tools.functions import retry_on_failure
from tools.predefined_category import get_entity_relations_by_domain
from tools.mapping import relation_match_scoring  # 导入从mapping模块中的函数

class EntityRelationStore:
    def __init__(self):
        self.relations = {}  

class EntityRelationEvaluator:

    def __init__(self, client, eval_model, log_file):
        self.client = client
        self.eval_model = eval_model
        self.log_file = log_file
        self.entity_relation_store = EntityRelationStore()

    def log_message(self, message: str, log_file: str = None):
        """Log a message with timestamp to console and optionally to a file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_msg + '\n')

    @retry_on_failure(max_retries=3)
    def extract_entities_relations(self, sent1: str, sent2: str, orig_idx: int, para_idx: int, orig_entities: List[Dict], para_entities: List[Dict], text_domain: str = None) -> None:
        # --- Relation Extraction ---
        if text_domain:
            self.log_message(f"Extracting relations for sentences [{orig_idx}]-[{para_idx}] using domain: {text_domain}", self.log_file)
        else:
            self.log_message(f"Extracting relations for sentences [{orig_idx}]-[{para_idx}] using default relation types", self.log_file)
            
        try:
            actual_orig_entities_json = json.dumps(orig_entities, ensure_ascii=False)
            actual_para_entities_json = json.dumps(para_entities, ensure_ascii=False)
            self.log_message(f"actual_orig_entities_json: {actual_orig_entities_json}", self.log_file)
            self.log_message(f"actual_para_entities_json: {actual_para_entities_json}", self.log_file)
            
        except Exception as dump_e:
            actual_orig_entities_json = "[]" # Default to empty JSON array string on error
            actual_para_entities_json = "[]"

        # Get domain-specific entity relation definitions if domain is provided
        if text_domain and text_domain in ["academic_engineering", "finance_business", "education", "literature_art", "politics_law"]:
            relation_types = get_entity_relations_by_domain(text_domain)
            domain_relation_dict = {relation["category"]: relation["definition"] for relation in relation_types}
            formatted_relation_definitions = "\n".join([f"- {category}: {definition}" for category, definition in domain_relation_dict.items()])
            self.log_message(f"Using domain-specific relation types for domain: {text_domain}", self.log_file)
        else:
            # Use default relation definitions if no domain is specified or domain is invalid
            raise ValueError(f"Invalid domain: {text_domain}")
            self.log_message(f"Using default relation types (no valid domain specified)", self.log_file)

        relation_prompt_filled = f"""
            Objective: Extract and output entity relations from TWO input sentences. You MUST analyze BOTH sentences and output their relations in a single response. Your output MUST contain relations for BOTH sentences, not just one.

            Predefined Relation Taxonomy and Definitions:
            {formatted_relation_definitions}

            === SENTENCE PAIR TO ANALYZE ===
            First Sentence: "{sent1}"
            First Sentence Entities: {actual_orig_entities_json}

            Second Sentence: "{sent2}"
            Second Sentence Entities: {actual_para_entities_json}

            Processing Instructions:
            1.  **Independent Analysis:** Process Sentence 1 (using its text and its Detected Entities list) and Sentence 2 (using its text and its Detected Entities list) entirely separately. You MUST analyze BOTH sentences and provide relations for each.
            2.  **Intra-Sentence Relations Only:** Identify relationships ONLY between pairs of entities listed in the Detected Entities for the *same* sentence. Do not establish relations between entities from different sentences.
            3.  **Candidate Identification:** For each sentence, consider potential relationships between all valid pairs of entities provided for that sentence (e.g., Entity A <-> Entity B).
            4.  **Relation Classification:** Classify identified relationships using the types defined in the 'Predefined Relation Taxonomy and Definitions' section above. Carefully consider the definition of each relation type. The classification should be based on the most evident and fitting relation according to the definition and sentence context.
            5.  **Strict Rule Adherence:** Apply validation rules meticulously:
                * **Entity Matching:** The `subject` and `object` values in the output relation MUST exactly match the `text` field of entities present in the corresponding input `Detected Entities` list.
                * **Type Constraints:** Ensure the nature of the subject and object entities aligns with the definition of the chosen relation type.
                * **Others Category:** Use the "Others" category ONLY as a last resort when none of the other predefined relation types clearly fit the relationship between entities. You must exhaustively consider all other relation types before classifying a relationship as "Others".
                * **All Relationships Required:** All detected entity pairs that have a relationship should have a relation output, even if the relation type is "Others".
            6.  **Completeness Check:** Before finalizing output, verify that:
                - Both sentences have been analyzed
                - All possible entity pairs within each sentence have been considered
                - Each identified relation has been properly classified
                - Your output MUST contain TWO objects, one for each sentence
                - Each sentence's relations must be analyzed with equal thoroughness

            Output Structure:
            {{
                "result": [
                    {{
                        "sentence_index": 0,
                        "original_sentence": "First Sentence Text",
                        "relations": [
                            {{
                                "subject": "entity1_exact_text",
                                "relation": "RELATION_TYPE",
                                "object": "entity2_exact_text"
                            }}
                        ]
                    }},
                    {{
                        "sentence_index": 1,
                        "original_sentence": "Second Sentence Text",
                        "relations": [
                            {{
                                "subject": "entity1_exact_text",
                                "relation": "RELATION_TYPE",
                                "object": "entity2_exact_text"
                            }}
                        ]
                    }}
                ]
            }}

            Example Input:
            First Sentence: "Tim Cook leads Apple, which is headquartered in Cupertino."
            Second Sentence: "The iPhone 15 was created by Apple and includes the A17 chip."

            Example Output:
            {{
                "result": [
                    {{
                        "sentence_index": 0,
                        "original_sentence": "Tim Cook leads Apple, which is headquartered in Cupertino.",
                        "relations": [
                            {{
                                "subject": "Tim Cook",
                                "relation": "Hierarchical",
                                "object": "Apple"
                            }},
                            {{
                                "subject": "Apple",
                                "relation": "Spatial",
                                "object": "Cupertino"
                            }}
                        ]
                    }},
                    {{
                        "sentence_index": 1,
                        "original_sentence": "The iPhone 15 was created by Apple and includes the A17 chip.",
                        "relations": [
                            {{
                                "subject": "Apple",
                                "relation": "Derivational",
                                "object": "iPhone 15"
                            }},
                            {{
                                "subject": "A17 chip",
                                "relation": "Structural",
                                "object": "iPhone 15"
                            }}
                        ]
                    }}
                ]
            }}

            FINAL CHECK:
            - Have you analyzed BOTH sentences?
            - Does your output contain EXACTLY TWO objects in the result array?
            - Are all entity texts exact matches?
            - Have you considered all possible entity pairs?
            - Have you applied the same analysis depth to both sentences?
            - Is your output wrapped in a JSON object with a "result" field containing the array?

            Now, analyze the provided sentence pair and output their relations following this format exactly.
        """
        self.log_message(f"relation_prompt_filled: {relation_prompt_filled}", self.log_file)
        try:
            rel_resp = self.client.chat.completions.create(
                model=self.eval_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a precise relation extraction system that ALWAYS analyzes BOTH input sentences.
Your task is to extract entity relations from TWO input sentences. You must:
1. Process BOTH sentences independently
2. Extract ALL possible relations between entities within EACH sentence
3. Return a JSON object with a "result" field containing an array of EXACTLY TWO objects
4. Never skip or ignore either sentence
5. Apply the same level of analysis to both sentences
6. Ensure each output object has the correct sentence_index (0 for first, 1 for second)
7. Follow the relation taxonomy strictly as provided
8. Return relations even if they seem obvious or trivial
Remember: Your output must be a JSON object with format: {"result": [sentence1_obj, sentence2_obj]}"""
                    },
                    {
                        "role": "user", 
                        "content": relation_prompt_filled
                    }
                ],
                temperature=0.5,
                response_format={"type": "json_object"}
            )

            raw_content = rel_resp.choices[0].message.content.strip()
            print(f"raw_content: {raw_content}")
            self.log_message(f"raw_content: {raw_content}", self.log_file)
            parsed_data = json_repair.loads(raw_content)
            print(f"parsed_data: {parsed_data}")
            self.log_message(f"parsed_data: {parsed_data}", self.log_file)

            # Handle different JSON output formats
            if isinstance(parsed_data, dict) and "result" in parsed_data:
                # If output is a JSON object with a "result" key containing a list
                result_list = parsed_data["result"]
                if len(result_list) >= 2:
                    orig_relations = result_list[0].get("relations", [])
                    para_relations = result_list[1].get("relations", [])
            elif isinstance(parsed_data, list):
                if len(parsed_data) >= 2:
                    # Output is already a list with at least 2 items
                    orig_relations = parsed_data[0].get("relations", [])
                    para_relations = parsed_data[1].get("relations", [])
            elif isinstance(parsed_data, dict) and len(parsed_data) >= 2:

                result_list = list(parsed_data.values())
                orig_relations = result_list[0].get("relations", []) if isinstance(result_list[0], dict) else []
                para_relations = result_list[1].get("relations", []) if isinstance(result_list[1], dict) and len(result_list) > 1 else []
            else:
                print(f"Warning: Unexpected structure in parsed relation data for indices {orig_idx}, {para_idx}: {parsed_data}")


            self.log_message(f"orig_relations in extract_entities_relations: {orig_relations}", self.log_file)
            self.log_message(f"para_relations in extract_entities_relations: {para_relations}", self.log_file)
            self.entity_relation_store.relations.setdefault('orig', []).append({orig_idx: orig_relations})
            self.entity_relation_store.relations.setdefault('para', []).append({para_idx: para_relations})

        except Exception as e:
            self.log_message(f"Error in relation extraction: {str(e)}", self.log_file)
            self.entity_relation_store.relations.setdefault('orig', []).append({orig_idx: []})
            self.entity_relation_store.relations.setdefault('para', []).append({para_idx: []})

    @retry_on_failure(max_retries=3)
    def _relation_match_score(self, orig_sent: str, para_sent: str, orig: list, para: list) -> float:
        return relation_match_scoring(
            self.client,
            self.eval_model,
            orig_sent,
            para_sent,
            orig,
            para,
            log_message=self.log_message,
            log_file=self.log_file
        )
 
    @retry_on_failure(max_retries=3)
    def evaluate_pair(self, alignment, sents, entities, text_domain: str = None) -> float:
        total_score = 0.0
        
        orig_other_idx = 0
        para_other_idx = 0
        # Process each aligned sentence pair
        for para_idx, orig_idx in alignment.items():
            try:
                # Safe type conversion with error handling
                try:
                    if orig_idx is not None:
                        orig_idx = int(orig_idx)
                    if para_idx is not None:
                        para_idx = int(para_idx)
                except (ValueError, TypeError) as e:
                    self.log_message(f"Error converting indices to integers: {str(e)}", self.log_file)
                    continue
     
                orig_sent, para_sent = sents[para_idx]["original"], sents[para_idx]["paraphrase"]
                
                orig_entities, para_entities = entities[para_idx]["original"], entities[para_idx]["paraphrase"]
                self.log_message(f"orig_sent: {orig_sent}", self.log_file)
                self.log_message(f"para_sent: {para_sent}", self.log_file)
                self.log_message(f"orig_entities: {orig_entities}", self.log_file)
                self.log_message(f"para_entities: {para_entities}", self.log_file)
                # Extract entities and relations from both sentences
                self.extract_entities_relations(orig_sent, para_sent, orig_idx, para_idx, orig_entities, para_entities, text_domain)          

                # Retrieve stored relations
                orig_rels = self.entity_relation_store.relations['orig'][orig_other_idx].get(orig_idx, [])
                orig_other_idx += 1
                para_rels = self.entity_relation_store.relations['para'][para_other_idx].get(para_idx, [])
                para_other_idx += 1

                self.log_message(f"Relation matching for indices [{orig_idx}]-[{para_idx}]:\n", self.log_file)

                self.log_message(f"  Original relations:\n", self.log_file)
                self.log_message(f"orig_rels: {orig_rels}", self.log_file)
                if orig_rels is None:
                    self.log_message(f"    - No relations detected.\n", self.log_file)
                else:
                    for rel in orig_rels:
                        self.log_message(f"    - {rel}\n", self.log_file)

                self.log_message(f"  Paraphrased relations:\n", self.log_file)
                self.log_message(f"para_rels: {para_rels}", self.log_file)
                if para_rels is None:
                    self.log_message(f"    - No relations detected.\n", self.log_file)
                else:
                    for rel in para_rels:
                        self.log_message(f"    - {rel}\n", self.log_file)
                
                # Calculate relation matching score
                rel_score = self._relation_match_score(orig_sent, para_sent, orig_rels, para_rels)

                self.log_message(f"  Relation matching score: {rel_score:.2f}\n", self.log_file)
                # Combined weighted score
                sentence_score = rel_score
                total_score += sentence_score
                
            except Exception as e:
                self.log_message(f"Error processing sentence pair [{orig_idx}]-[{para_idx}]: {str(e)}\n", self.log_file)
                continue

        # Calculate final average score (0 if no alignment)
        final_score = total_score / len(alignment) if alignment else 0.0
        self.log_message(f"Final entity relation evaluation score: {final_score:.2f}\n", self.log_file)
        self.log_message("="*50 + "\n", self.log_file)

        return final_score
