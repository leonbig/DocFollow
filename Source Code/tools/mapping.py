from typing import List, Dict
from textwrap import dedent
import json_repair

def direct_sentence_matching_llm(client, eval_model, source: List[str], target: List[str]) -> Dict[int, int]:
    matches = {}
    for i, target_sent in enumerate(target):
        start = max(0, i - 3)
        end = min(min(len(target), len(source)), i + 4)
        if start > end:
            candidates = []
        candidates = source[start:end]
        candidate_text = "\n".join([f"{idx}: {sent}" for idx, sent in enumerate(candidates)])
        prompt = f'''Task: Find the corresponding rewritten sentence from candidates for the given source sentence.

        Source sentence:
        "{target_sent}"

        Candidate sentences:
        {candidate_text}

        Matching Criteria:
        Compare the source sentence with each candidate, select the one that best matches the source sentence.

        Important Note:
        The topics may be different, focus on how information is presented rather than semantic similarity.

        Output Instructions:
        - If a corresponding text is found: return ONLY its index number (0-based)
        - If no corresponding text exists: return ONLY "No-Match"
        - Do not include any explanations or additional text

        Response format:
        - Match found: "0" or "1" or "2" etc.
        - No match: "No-Match"'''
        
        response = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        chosen = response.choices[0].message.content.strip()
        try:
            idx = int(chosen)
            if 0 <= idx < len(candidates):  # 确保索引在有效范围内
                matches[i] = start + idx
        except ValueError:
            matches[i] = None # 标记为无效匹配
    return matches

def direct_sentence_matching_one2one(source: List[str], target: List[str]) -> Dict[int, int]:
    min_length = min(len(source), len(target))
    matches = {}
    for i in range(min_length):
        matches[i] = i

    return matches

def entity_match_scoring(client, eval_model, orig_sent, para_sent, orig_entities: List[Dict], para_entities: List[Dict], log_message=None, log_file=None) -> float:

    # Default logging function if none provided
    if log_message is None:
        log_message = lambda msg, file=None: None
    
    # Skip scoring if no original entities or entity count difference >30%
    n_orig = len(orig_entities)
    n_para = len(para_entities)

    log_message(f"\nEntity matching details:", log_file)
    log_message(f"Original entities count: {n_orig}", log_file)
    log_message(f"Paraphrased entities count: {n_para}", log_file)

    # Skip if no original entities
    if n_orig == 0:
        log_message("No original entities found, score = 0.0", log_file)
        return 0.0
    if n_para == 0:
        log_message("No paraphrase entities found, score = 0.0", log_file)
        return 0.0

    # Compute expected mapping size
    L = min(n_orig, n_para)
    log_message(f"Computing mapping for {L} entities", log_file)

    # Prepare LLM prompt for entity mapping
    prompt = dedent(f"""
        You are an expert in entity matching.
        Map each original entity index to its best matching paraphrased entity index.
        Return ONLY a JSON object mapping original indices to paraphrase indices with exactly {L} entries.

        Original sentence: {orig_sent}
        Original entities (with indices):
        {[{"index": i, "text": e['text']} for i, e in enumerate(orig_entities)]}

        Paraphrased sentence: {para_sent}
        Paraphrased entities (with indices):
        {[{"index": i, "text": e['text']} for i, e in enumerate(para_entities)]}

        Required output format:
        {{
            "0": 2,  // means paraphrased entity at index 0 matches original entity at index 2
            "1": 0,  // means paraphrased entity at index 1 matches original entity at index 0
            // ... more mappings
        }}
        
        Important:
        - Use ONLY the indices (numbers) for mapping, not the text values
        - Ensure the mapping is 1-to-1 and no entity is unmatched
        - Return ONLY the mapping object, no explanations
        """)
    try:
        resp = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        raw_content = resp.choices[0].message.content.strip()
        clean_content = json_repair.repair_json(raw_content)
        raw_mapping = json_repair.loads(clean_content)
        mapping = {int(k): int(v) for k, v in raw_mapping.items()}

        log_message("\nEntity mapping results:", log_file)
        # Count matches of type-compatible entity pairs
        match_count = 0
        for p_idx, o_idx in mapping.items():
            if 0 <= o_idx < len(orig_entities) and 0 <= p_idx < len(para_entities):
                orig_entity = orig_entities[o_idx]
                para_entity = para_entities[p_idx]
                log_message(f"\nMapping: Original[{o_idx}] -> Paraphrase[{p_idx}]", log_file)
                log_message(f"  Original: {orig_entity['text']} ({orig_entity['type']})", log_file)
                log_message(f"  Paraphrase: {para_entity['text']} ({para_entity['type']})", log_file)
                
                # 实体类型匹配判断 - 包括当两个实体类型都是OTHERS时也视为匹配
                if orig_entity["type"] == para_entity["type"]:
                    match_count += 1
                    log_message(f"  ✓ Types match: {orig_entity['type']}", log_file)
                else:
                    log_message(f"  ✗ Type mismatch: original: {orig_entity['type']} vs paraphrase: {para_entity['type']}", log_file)

        score = min(1.0, match_count / len(para_entities))
        log_message(f"\nMatched pairs: {match_count}/{len(para_entities)}", log_file)
        log_message(f"Final entity match score: {score:.2f}", log_file)
        return score

    except Exception as e:
        log_message(f"Error in entity matching: {str(e)}", log_file)
        return 0.0


def relation_match_scoring(client, eval_model, orig_sent: str, para_sent: str, orig: list, para: list, log_message=None, log_file=None) -> float:

    # Default logging function if none provided
    if log_message is None:
        log_message = lambda msg, file=None: None
        
    try:
        n_orig = len(orig)
        n_para = len(para)
        
        if n_orig == 0:
            log_message(f"No original relations, score=0.0", log_file)
            return 0.0
        if n_para == 0:
            log_message(f"No paraphrase relations, score=0.0", log_file)
            return 0.0
            
        orig_triples = [(r['subject'], r['relation'], r['object']) for r in orig]
        para_triples = [(r['subject'], r['relation'], r['object']) for r in para]
        L = min(n_orig, n_para)
        
        if L == 0:
            log_message("No valid relations to compare, score=0.0", log_file)
            return 0.0
            
        triple_strings_orig = [f"({s} -[{rel}]-> {o})" for s, rel, o in orig_triples]
        triple_strings_para = [f"({s} -[{rel}]-> {o})" for s, rel, o in para_triples]

        log_message(f"Relation matching for:\n  Original: {orig_sent}\n  Paraphrase: {para_sent}\n", log_file)
        log_message(f"  Original triples ({len(orig_triples)}): {triple_strings_orig}\n", log_file)
        log_message(f"  Paraphrased triples ({len(para_triples)}): {triple_strings_para}\n", log_file)

        prompt = dedent(f"""
        You are an expert in relation matching.
        Map each original relation index to its best matching paraphrased relation index.
        Return ONLY a JSON object mapping original indices to paraphrase indices with exactly {L} entries.

        Original sentence: {orig_sent}
        Original relation triples with indices:
        {[{"index": i, "triple": triple} for i, triple in enumerate(triple_strings_orig)]}

        Paraphrased sentence: {para_sent}
        Paraphrased relation triples with indices:
        {[{"index": i, "triple": triple} for i, triple in enumerate(triple_strings_para)]}

        Required output format:
        {{
            "0": 2,  // means paraphrased relation at index 0 matches original relation at index 2
            "1": 0,  // means paraphrased relation at index 1 matches original relation at index 0
            // ... more mappings up to {L} entries total
        }}

        Important rules:
        1. Output ONLY the index mapping JSON object
        2. Each original index must map to exactly one paraphrase index
        3. Use ONLY numerical indices (0, 1, 2, etc.)
        4. Include exactly {L} mappings
        5. Ensure all indices are valid (within range of the provided triples)
        6. If both relations are classified as "Others", consider it a valid match too - this means relationships where both original and paraphrase use "Others" classification should be counted as matching
        """)

        response = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        raw_content = response.choices[0].message.content.strip()
        raw_mapping = json_repair.loads(raw_content)
        mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        if len(mapping) != L:
            raise ValueError(f"Expected mapping of size {L}, got {len(mapping)}")

        match_count = 0
        log_message(f"  Relation matching results", log_file)

        for p_idx, o_idx in mapping.items():
            if 0 <= o_idx < n_orig and 0 <= p_idx < n_para:
                orig_rel = orig_triples[o_idx]
                para_rel = para_triples[p_idx]
                # Consider a match either if relations are identical OR both are "Others"
                match = orig_rel[1] == para_rel[1]
                log_message(f"    Original[{o_idx}] {orig_rel} <-> Paraphrase[{p_idx}] {para_rel} | ", log_file)
                log_message(f"✓ Match\n" if match else "✗ Mismatch\n", log_file)
                if orig_rel[1] == para_rel[1]:
                    match_count += 1
                    
        if match_count < 0:
            log_message("Warning: Negative match count detected, setting to 0", log_file)
            match_count = 0
            
        score = min(1.0, match_count / n_para)
            
        log_message(f"  Matched pairs: {match_count}/{n_para}", log_file)
        log_message(f"  Final relation match score: {score:.2f}", log_file)
        return score
        
    except Exception as e:
        log_message(f"Error in relation matching: {str(e)}", log_file)
        return 0.0
    



def classify_text_domain(client, eval_model, text: str, log_message=None, log_file=None) -> str:

    if log_message is None:
        log_message = lambda msg, file=None: None
        
    domain_prompt = f"""
    Task: Analyze the given text and classify it into the most appropriate domain category.
    
    Available domains with detailed descriptions:
    
    1. academic_engineering
       Description: Scientific or technical content related to research, academia, or engineering fields
       Keywords: research, methodology, theory, experiment, technology, engineering, scientific, innovation, academia, hypothesis
       Examples: Research papers, technical documentation, scientific articles, engineering specifications
    
    2. finance_business
       Description: Content related to finance, economics, business operations, markets, or commerce
       Keywords: market, profit, investment, business, company, finance, economic, trade, commercial, stock
       Examples: Financial reports, business news, economic analyses, market trends, commercial agreements
    
    3. education
       Description: Content focused on teaching, learning, educational institutions, or pedagogical approaches
       Keywords: learning, teaching, education, student, curriculum, pedagogy, school, assessment, instruction, course
       Examples: Educational materials, teaching guidelines, school policies, learning assessments, academic programs
    
    4. literature_art
       Description: Content about creative works, artistic expression, cultural topics, or literary analysis
       Keywords: culture, art, literature, creative, artistic, novel, painting, music, poetry, performance
       Examples: Literary criticism, artistic reviews, cultural analyses, creative works, artistic interpretations
    
    5. politics_law
       Description: Content concerning governance, legal frameworks, political systems, or civic matters
       Keywords: government, law, policy, regulation, political, legal, court, legislation, rights, constitution
       Examples: Legal documents, political analyses, policy papers, governmental reports, constitutional texts
    
    Input text: "{text[:1000]}..." (truncated for brevity)
    
    Instructions:
    1. Analyze the key terminology, context, and subject matter of the text
    2. Consider the primary focus and purpose of the content
    3. Determine which domain best captures the main subject area
    4. Return ONLY one of the following domain identifiers as your response:
       - academic_engineering
       - finance_business
       - education
       - literature_art
       - politics_law
    5. Do not include any other text, explanation, or formatting in your response
    """
    
    try:
        domain_resp = client.chat.completions.create(
            model=eval_model,
            messages=[{"role": "user", "content": domain_prompt}],
            temperature=0.0
        )
        
        # 提取并清理域名
        raw_content = domain_resp.choices[0].message.content.strip().lower()
        
        # 验证域名有效性
        valid_domains = ["academic_engineering", "finance_business", "education", "literature_art", "politics_law"]
        for valid_domain in valid_domains:
            if valid_domain in raw_content:
                log_message(f"Text classified as domain: {valid_domain}", log_file)
                return valid_domain
        
        # 如果无法匹配精确的域名，检查是否包含部分关键词
        domain_keywords = {
            "academic_engineering": ["academic", "engineering", "scientific", "research", "technology"],
            "finance_business": ["finance", "business", "economic", "market", "commercial"],
            "education": ["education", "learning", "teaching", "school", "student"],
            "literature_art": ["literature", "art", "creative", "culture", "artistic"],
            "politics_law": ["politics", "law", "government", "legal", "policy"]
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in raw_content:
                    log_message(f"Domain classified based on keyword '{keyword}': {domain}", log_file)
                    return domain
        
        # 回退到默认域
        log_message("Could not determine specific domain, using academic_engineering as default", log_file)
        return "academic_engineering"
        
    except Exception as e:
        log_message(f"Error during domain classification: {str(e)}, using default domain", log_file)
        return "academic_engineering"  