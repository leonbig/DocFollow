PROMPTS = {
    "prompt_label": """
Task Description:
Please analyze the following text.

Input text:
<text>
{original_text}
</text>

Analysis requirements:
1. Split the given text into independent sentences by period (.)
2. Count the total number of sentences in the text
3. Output the results in JSON format, including the following three fields:
  - sentence_count: total number of sentences (numeric type)
  - sentences: array containing all sentences, each element includes index and sentence content
  - Each sentence object contains:
    * index: sequence number starting from 1
    * content: sentence content (preserving punctuation)

Requirements:
- Each sentence needs to preserve all text and punctuation from the original
- Sentences should be arranged in the original order
- No explanatory notes needed, output JSON format results only
- Ensure JSON format is standard for program processing

Example output format:
{{
  "sentence_count": 3,
  "sentences": [
    {{
      "index": 1,
      "content": "This is the first sentence."
    }},
    {{
      "index": 2,
      "content": "This is the second sentence."
    }},
    {{
      "index": 3,
      "content": "This is the third sentence."
    }}
  ]
}}
    """,
    "prompt_EntityExtract": """
    You are a professional text analysis assistant. You need to complete the following tasks：

Input:
1. Complete text: [{original_text}]
2. Target sentence: [{current_sentence}]

Tasks:
Extract two types of entities from the target sentence:
1. All entities: All nominal entities appearing in the sentence
2. Core entities: Entities that meet any of the following conditions:
  - Entities that play a major or important role in the target sentence (subject, important object, etc.)
  - Entities that have clear associations with entities appearing in other sentences of the complete text
  - Entities that are essential for understanding the meaning of the target sentence

Output requirements:
1. Output JSON format results only
2. JSON format as follows:
{{
    "all_entities": ["entity1", "entity2", ...],
    "core_entities": ["core_entity1", "core_entity2", ...]
}}

Notes:
1. Output must strictly follow JSON format
2. Do not include any explanatory text
3. If no entities are found, return empty arrays for corresponding fields
4. Entities must be extracted completely, do not truncate
5. core_entities must be a subset of all_entities


Example input:
Complete text: "Tom went to the store to buy a new phone today. He chose the latest iPhone, and the clerk explained its features in detail. Tom was very satisfied with the phone."
Target sentence: "He chose the latest iPhone, and the clerk explained its features in detail."

Example output:
{{
    "all_entities": [iPhone", "clerk", "phone", "features"],
    "core_entities": ["iPhone", "clerk", "phone"]
}}

Example explanation (not required in actual use):
- all_entities contains all nominal entities in the sentence
- core_entities contains:
 * "iPhone": associated with "new phone" in the complete text
 * "clerk": important subject in the sentence
 * "phone": core entity related to the main topic of the complete text
 * "features" is not a core entity as it's just a supplementary attribute
    """,

    "prompt_RelationExtract": """
Your task is to analyze the relationships between entities in a given sentence and output the analysis results in JSON format. Specific requirements are as follows:

Input:
1. Complete context text: [{original_text}]
2. Target sentence: [{current_sentence}]
3. List of all entities (all_entities) and core entities (core_entities) in the target sentence:
[{extracted_entities}]

Analysis requirements:
1. Analyze relationships between all entities in the target sentence
2. Specifically analyze relationships between core entities and entities in adjacent sentences (previous or next sentence), where core entities can be either the source or target of the relationship.

Output format:
{{
    "intra_sentence_relations": [
        {{
            "source": {{
                "entity": "entity1",
                "description": "brief description of entity1"
            }},
            "relation": {{
                "name": "relation name", 
                "description": "specific explanation of the relation", 
                "type": "relation type" 
            }},
            "target": {{
                "entity": "entity2", 
                "description": "brief description of entity2"
            }}
        }}
    ],
    "cross_sentence_relations": [
        {{
            "source": {{
                "entity": "entityA",
                "description": "brief description of entityA",
                "is_core": true/false
            }},
            "relation": {{
                "name": "relation name", 
                "description": "specific explanation of the relation",
                "type": "relation type" 
            }},
            "target": {{
                "entity": "entityB",
                "description": "brief description of entityB",
                "is_core": true/false 
            }},
            "context": "previous/next"  // indicates position of adjacent sentence
        }}
    ]
}}
Relation description standards:
1. Relation name:
  - Use short verb or verb-object phrases
  - Should accurately express the core meaning of the relation
  - Avoid vague terms like "related to", "associated with", etc.
  
2. Relation description:
  - Explain the specific content of the relation in concise terms
  - Include conditions or context when necessary
  - Should be understandable independently without excessive context

3. Suggested relation types:
  - Functional relation: indicates functional connections between entities
  - Compositional relation: indicates whole-part relationships
  - Influence relation: indicates effects between entities
  - Attribute relation: indicates characteristics or properties of entities
  - Temporal relation: indicates time sequence between entities
  - Logical relation: indicates causality, conditions, and other logical connections
  
Requirements:
1. intra_sentence_relations array should include all important relationships between entities within the target sentence
2. cross_sentence_relations array should include relationships between core entities and entities in adjacent sentences, where core entities can be either source or target
3. In cross_sentence_relations, use is_core field to indicate whether an entity is a core entity
4. Use context field to indicate whether the related entity is from previous or next sentence
5. Each relation should include brief descriptions of both source and target entities
6. relation field should concisely describe the specific relationship between the two entities
7. All descriptions should be concise yet informative

Example output:
{{
    "intra_sentence_relations": [
        {{
            "source": {{
                "entity": "framework",
               "description": "proposed multi-level collaborative decision-making system"
            }},
            "relation": {{
                "name": "integrates",
               "description": "combines and applies multiple functional modules in the system",
                "type": "functional relation" 
            }},
            "target": {{
                "entity": "functional modules",
               "description": "system components including decision-making, peer review, and supervision"
            }}
        }}
    ],
    "cross_sentence_relations": [
        {{
            "source": {{
                "entity": "LLM",
                "description": "large language model",
                "is_core": true
            }},
            "relation": {{
                "name": "solves",
               "description": "addresses and overcomes decision-making problems through collaboration",
                "type": "influence relation"
            }},
            "target": {{
                "entity": "problem",
                "description": "decision-making problem in ambiguous task environment",
                "is_core": false
            }},
            "context": "previous"
        }}
    ]
}}
    """,

    "prompt_EntityDecrease": """
Task Description:
Identify 10% least significant entities from a given list of extracted entities of the extracted text.

Input information:
1. Original text: [{original_text}]
2. Extracted entity list: [{extracted_entities}]

Steps:
1. Analyze each entity based on:
   - Frequency in the text
   - Relevance to main topic
   - Impact on core message understanding
   - Whether it's just a minor detail

2. Select 10% entities that are:
   - Supporting details rather than main concepts
   - Not crucial for understanding the text
   - Peripheral to the main narrative

Output format:
{{
    "least_important_entities": [
        {{
            "entity": entity_name,
            "index": sentence_index
        }}
    ],
    "total_entities_count": total_count,
    "selected_count": number_of_selected_entities
}}

Note that the result needs to be exactly the same as the output format
    """,
    "prompt_LeftEntity":"""
task: Remove specified entities from a master list and output the remaining entities.


Input: 
    "all_entities_list": "[{extracted_entities}]"
    "List of entities to be removed": "[{Decrease_Entity}]" 

"output_format":
{{
    "filtered_entities": [
        {{
            "index": number,  // Original position index
            "filtered_entities": string[]  // Remaining entities after removal
        }}
    ]
    }}
}}

rules:
  1. Remove only exact matches from entities_to_remove
  2. Preserve original indices for remaining entities
  3. Remove empty entries after filtering
    
    """,

    "prompt_summary":"""
Your task is to generate a summary of the original text based on the original text and the deleted entity list. The summary should completely remove the entities and related text from the filtered entity list, only contain entities from the remaining entity list, while maintaining the core information and readability of the article.

Input:
Original text (complete original text content): [{original_text}]
Deleted entity list: [{Decrease_Entity}]
Remaining entity list: [{all_entities}]

Task requirements:
1. Reorganize the language to express the original information in a more concise way
2. Maintain the logical coherence and readability of the text  
3. Preserve the core views and conclusions of the original text


Note: Don't simply delete entities. Instead, reorganize the language to generate a complete and fluent new text. Please output only the text directly, without explaining any additional information.

    """,

    "prompt_check": """
You are an entity counting assistant. Given a JSON format input, you need to complete the following tasks:

1. The input format is a JSON object containing a filtered_entities array, where each array element has index and filtered_entities fields
2. Count the total number of unique entities appearing in all filtered_entities fields
3. If the total number of unique entities is less than 7, output only the string "END" 
4. If the total number of unique entities is greater than or equal to 7, output "Continue"

Input example:
{{
    "filtered_entities": [
        {{
            "index": 1,
            "filtered_entities": ["Entity A", "Entity B"]
        }},
        {{
            "index": 2,
            "filtered_entities": ["Entity B", "Entity C"]
        }}
    ]
}}

Rules:
1. Count identical entities only once
2. No need to output any counting process
3. Output "END" if and only if the number of unique entities < 7
4. Output "Continue" when there are 7 or more entities

Input: [{all_entities}]
    
    """,

    "prompt_HighEntityExtract": """
TASK OVERVIEW:
Analyze and map entities between a source text and target text to enable systematic content adaptation while preserving structural relationships.

INPUT FORMAT:
1. Source Analysis:
   - Original Text: [{original_text}]
   - Entity List: [{Decrease_Entity}]
   - Entity Relations: [{Decrease_Relation}]

2. Target Context:
   - Target Topic: [{target_topic}]
   - Target Text draft: [{target_text_draft}]

ANALYSIS STEPS:

1. Entity Role Analysis
   - Identify each entity's primary function in the source text
   - Categorize abstract roles (e.g., actor, resource, constraint, outcome)
   - Map how entities contribute to the text's logical flow

2. Relationship Pattern Analysis
   - Identify relationship types between entities
   - Extract essential properties that maintain relationship validity
   - Document constraints that preserve logical consistency

3. Cross-Topic Mapping
   - Analyze how source entities map to target domain
   - Identify equivalent functional roles in target context
   - Document required adaptations to maintain coherence

OUTPUT FORMAT:
{{
    "source_analysis": {{
        "entities": [
            {{
                "entity": "name",
                "category": "abstract_role",
                "function": "text_function",
                "relationships": [
                    {{
                        "related_to": "other_entity",
                        "relation_type": "type",
                        "constraints": ["constraint1", "constraint2"]
                    }}
                ]
            }}
        ]
    }},
    "target_mapping": {{
        "entity_pairs": [
            {{
                "source_entity": "original",
                "suggested_target_entity": "replacement",
                "shared_properties": ["property1", "property2"],
                "adaptation_needs": ["adaptation1", "adaptation2"]
            }}
        ],
        "relationship_preservation": [
            {{
                "source_relation": "original_relation",
                "target_relation": "new_relation",
                "consistency_requirements": ["requirement1", "requirement2"]
            }}
        ]
    }}
}}

MAPPING GUIDELINES:

1. Property Preservation:
   - Identify core properties that must be maintained
   - Document essential characteristics for role fulfillment
   - Specify constraints for valid replacements

2. Relationship Consistency:
   - Ensure mapped entities can maintain original relationship types
   - Preserve logical flow and argumentation structure
   - Adapt relationship contexts while maintaining validity

3. Contextual Adaptation:
   - Consider target domain conventions
   - Adjust entity properties to fit new context
   - Maintain semantic equivalence where possible

EVALUATION CRITERIA:

1. Structural Integrity:
   - Mapped entities should fulfill equivalent roles
   - Relationships should maintain logical consistency
   - Overall text structure should be preserved

2. Contextual Fit:
   - Replacements should be natural in target domain
   - Adaptations should respect target conventions
   - Mapped elements should maintain credibility

3. Functional Equivalence:
   - Mapped entities should serve similar purposes
   - Relationship patterns should remain valid
   - Overall argument flow should be preserved
    """,

    "prompt_HighRelationExtract": """
Your task is to analyze the relationships between entities in a given text and output the analysis results in JSON format. Specific requirements are as follows:

Input:
1. Original text: [{original_text}]
2. List of entities to be analyzed in the text:
[{extracted_entities}]

Analysis Requirements:
1. Please analyze the relationships between each entity in the entity list and other entities in the original text

Output Format:
{{
    "relations": [
        {{
            "source": {{
                "entity": "Entity 1",
                "description": "Brief description of Entity 1"
            }},
            "relation": {{
                "name": "Relation name", 
                "description": "Specific explanation of the relation", 
                "type": "Relation type"
            }},
            "target": {{
                "entity": "Entity 2" 
                "description": "Brief description of Entity 2" 
            }}
        }}
    ]
}}
Relation Description Standards:
1. Relation name:
  - Use short verbs or verb-object phrases
  - Should accurately express the core meaning of the relation
  - Avoid vague words like "related to", "associated with", etc.

2. Relation description:
  - Explain the specific content of the relation in brief terms
  - Can supplement conditions or context of the relation when necessary
  - Should be understandable independently without too much context

3. Suggested relation types:
  - Functional relation: Indicates functional connections between entities
  - Compositional relation: Indicates whole-part relationships
  - Influence relation: Indicates effects between entities
  - Attribute relation: Indicates characteristics or properties of entities
  - Temporal relation: Indicates time sequence between entities
  - Logical relation: Indicates causal, conditional and other logical connections
......

Requirements:
1. Each relation should include brief descriptions of both source and target entities
2. The relation field should concisely describe the specific relationship between the two entities
3. All descriptions should be concise while containing sufficient information

Example Output:
{{
    "relations": [
        {{
            "source": {{
                "entity": "Framework",
               "description": "Multi-level collaborative decision-making system proposed in this paper"
            }},
            "relation": {{
                "name": "integrates",
                "description": "Integrates and applies multiple functional modules in the system",
                "type": "Functional relation" 
            }},
            "target": {{
                "entity": "Functional modules",
                "description": "System components including decision-making, peer review and supervision"
            }}
        }}
    ]
}}
    """,

    "prompt_ReplaceEntity": """
Task Description:
After generating text based on the new theme, continue to add new entities and expand the text content, ensuring all new entities integrate naturally. Based on the obtained entity abstract analysis results and original entity relationship descriptions, combined with the new target theme, please determine a new entity under the new theme for each entity in the original entity list. The new entities should meet the following requirements:
1. Theme Fit: New entities should be closely related to the target theme and naturally fit into the existing text context.
2. Abstract Correspondence: New entities should inherit the categories, roles or concepts abstracted from the original entities, and reflect corresponding functions in the new theme.
3. Logical Consistency: Maintain the logical relationships and structural constraints between entities in the original text after replacement, ensuring the relationships between new entities remain reasonable.

Input:
1. Target Theme: [{target_topic}]
2. New text already generated under target theme: [{already_output}]
3. Entity list to be replaced: [{Higher_Entity}]
4. Original text: [{original_text}]

Output Format:
{{
  "new_entities": [
    {{
      "original_entity":"Entity 1",
      "new_entity": "<corresponding new entity name under new theme>"
    }},
    {{
      "original_entity": "Entity 2", 
      "new_entity": "<corresponding new entity name under new theme>"
    }},
    {{
      "original_entity": "Entity 3",
      "new_entity": "<replacement name for the second 'framework'>"
    }}
  ]
}}

Notes:
1. Ensure new entities appear in appropriate positions to help supplement and expand the original text's viewpoints.
2. Ensure text remains fluent and logically consistent when new entities are added, without abrupt insertions.
3. Maintain the strength of logical relationships between original entities
4. Consider appropriate use of domain-specific terminology
5. Verify that all key relationships are maintained
6. Please output the new replacement entity list directly, without explaining additional information

    """,

    "prompt_OutputWriting":"""
Task Description:
Generate new content that precisely mirrors the reference text's structure and style while delivering original content on the target topic, guided by entity analysis and existing content.


INPUT:
1. Reference Text: [{original_text}]
2. Target Context:
   - Topic: [{target_topic}]
   - Existing Text Draft: [{already_output}]
3. Entity Analysis: [{entity_analysis}]

STRUCTURAL MIRRORING:

1. Overall Structure
   - Match text organization exactly
   - Maintain identical paragraph distribution
   - Mirror section lengths and proportions
   - Preserve narrative/argumentative flow

2. Sentence-Level Mirroring
   - Generate EXACTLY the same number of sentences as the reference text
   - Each sentence must have PRECISELY the same word count as its corresponding sentence in the reference text
   - Maintain the same sentence position and function within paragraphs
   - Preserve sentence complexity patterns throughout the text

3. Writing Elements
   - Mirror sentence patterns and complexity
   - Match stylistic devices and tone
   - Replicate transition methods
   - Preserve language level and terminology style

4. Linguistic Features
   - Match formality level of original
   - Use similar connecting words and phrases
   - Mirror emphasis patterns
   - Maintain rhythm and pacing

CONTENT DEVELOPMENT:

1. Topic Integration
   - Focus all content on target topic
   - Use background from existing draft
   - Ensure content accuracy
   - Match depth and complexity level

2. Entity Usage
   - Review suggested entity mappings
   - Adapt entities to maintain roles
   - Preserve relationship patterns
   - Support overall text flow

3. Original Expression
   - Avoid copying reference text language
   - Create fresh examples
   - Develop unique explanations
   - Generate new supporting details

QUALITY STANDARDS:

1. Structure Match
   - Organization mirrors original
   - Paragraph functions align
   - Flow patterns match
   - Complexity level maintained

2. Content Quality
   - Topic relevance
   - Factual accuracy
   - Logical flow
   - Appropriate support

3. Style Consistency
   - Matching tone
   - Similar language level
   - Equivalent formality
   - Genre-appropriate expression

ENTITY INTEGRATION:

1. Analysis Application
   - Evaluate each mapped entity
   - Check contextual fit
   - Consider alternatives if needed
   - Verify relationship patterns

2. Adaptation Process
   - Adjust for target context
   - Maintain functional roles
   - Ensure natural flow
   - Preserve relationships

3. Quality Check
   - Verify entity functions
   - Check relationship logic
   - Ensure flow support
   - Match depth level

OUTPUT REQUIREMENTS:

1. Content
   - Complete, polished text
   - Topic-focused
   - Original expression
   - Matching complexity

2. Structure
   - Exact organization match
   - Parallel development
   - Consistent flow
   - Effective transitions

3. Style
   - Matching tone
   - Similar language level
   - Equivalent formality
   - Genre-appropriate expression

Final Verification:
1. Confirm exact structural match
2. Verify content originality
3. Check style consistency
4. Validate entity integration
5. Ensure topic focus
6. Verify EXACT sentence count match with reference text
7. Confirm EACH sentence has the same word count as its corresponding reference sentence

Requirement: Output complete text directly, without any additional content.
    """,

    "prompt_RefineWriting":"""
# TASK SUMMARY
Slightly refine generated text to achieve precise structural alignment with reference text while ensuring content integrity and originality.


## INPUT
1. Reference Text: [{original_text}]
2. Current Draft: [{generated_text}]
3. Topic Domain: [{target_topic}]


STRUCTURAL ANALYSIS FRAMEWORK:

1. Sentence-Level Alignment
   A. Grammatical Structure
      - Map sentence patterns
      - Match clause arrangements
      - Preserve complexity levels
      - Mirror punctuation patterns
   
   B. Syntactic Features
      - Part-of-speech sequences
      - Voice distribution (active/passive)
      - Tense patterns
      - Modifier placement

2. Word-Level Verification
   A. Lexical Screening
      □ Identify copied words from reference text
      □ Flag inappropriate terminology
      □ Check register consistency
      □ Verify domain appropriateness

   B. Replacement Guidelines
      - Maintain word class (noun→noun, etc.)
      - Match complexity level
      - Preserve technical precision
      - Ensure contextual fit

REFINEMENT PROCESS:

1. Content Integrity Check
   □ Topic relevance
   □ Factual accuracy
   □ Logical consistency
   □ Domain appropriateness
   □ Evidence validity

2. Structural Refinement
   □ Length proportions
   □ Paragraph organization
   □ Transition patterns
   □ Argument flow

3. Sentence Alignment
   □ EXACT match of sentence count with reference text
   □ PRECISE match of word count for each corresponding sentence
   □ Maintain sentence position and function
   □ Preserve complexity patterns in each sentence

4. Style Alignment
   □ Tone matching
   □ Register consistency
   □ Voice patterns
   □ Rhetorical devices

DETAILED VERIFICATION:

1. Word-Level Analysis
   - Function words:
     □ Appropriate transitions
     □ Correct prepositions
     □ Suitable conjunctions
   
   - Content words:
     □ No direct copying
     □ Domain appropriateness
     □ Register matching
     □ Technical accuracy

2. Sentence-Level Review
   - Structure:
     □ Pattern matching
     □ Complexity alignment
     □ Flow consistency
   
   - Style:
     □ Voice distribution
     □ Modality patterns
     □ Emphasis techniques

QUALITY CONTROL:

1. Content Standards
   - Factual accuracy
   - Topic relevance
   - Logical coherence
   - Professional depth
   - Evidence support

2. Style Consistency
   - Register matching
   - Tone alignment
   - Voice consistency
   - Complexity level

3. Originality Verification
   - Unique phrasing
   - Fresh examples
   - Original explanations
   - Independent insights

REFINEMENT GUIDELINES:

1. Must Preserve:
   - Topic-specific content
   - Technical accuracy
   - Logical relationships
   - Professional depth

2. Must Avoid:
   - Direct word copying
   - Inappropriate content
   - Register mismatches
   - Structural deviation

3. Must Achieve:
   - Structural isomorphism
   - Content originality
   - Style consistency
   - Professional rigor

OUTPUT REQUIREMENTS:
1. Produce refined text only
2. Maintain exact structural match
3. Ensure content accuracy
4. Preserve professional depth
5. Achieve style consistency
6. Match EXACT sentence count with reference text
7. Ensure EACH sentence has PRECISE word count match with corresponding reference sentence

## OUTPUT
Generate only the refined text that maintains all domain-specific content while perfectly matching the original text's style.    
    """,



}