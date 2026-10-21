import os
import datetime
from typing import Optional, List
import spacy
from tools.functions import log_message

def calculate_sentence_score(original_sent_count: int, paraphrased_sent_count: int, log_file: Optional[str] = None) -> float:
    if original_sent_count == 0:
        log_message("Warning: Original text has 0 sentences", log_file)
        return 0.0
    
    # 根据仿写文本句子数与原文句子数的关系分为两种情况
    if paraphrased_sent_count <= original_sent_count:
        # 情况1: 仿写文本句子数少于或等于原文
        # 使用比例计算分数
        ratio = paraphrased_sent_count / original_sent_count
        score = ratio * 100
        log_message(f"Case: Paraphrased text has fewer or equal sentences", log_file)
        log_message(f"Ratio calculation: {paraphrased_sent_count}/{original_sent_count} = {ratio:.2f}", log_file)
    else:
        # 情况2: 仿写文本句子数多于原文
        # 计算额外句子的比例，并减少分数
        extra_ratio = (paraphrased_sent_count - original_sent_count) / original_sent_count
        # 限制惩罚上限，确保分数不会过低
        penalty = extra_ratio * 100
        score = 100 - penalty
        log_message(f"Case: Paraphrased text has more sentences", log_file)
        log_message(f"Extra ratio: ({paraphrased_sent_count}-{original_sent_count})/{original_sent_count} = {extra_ratio:.2f}", log_file)
        log_message(f"Penalty: min(50, {extra_ratio:.2f}*100) = {penalty:.2f}", log_file)
    
    log_message(f"Original sentence count: {original_sent_count}", log_file)
    log_message(f"Paraphrased sentence count: {paraphrased_sent_count}", log_file)
    log_message(f"Sentence count score: {score:.2f}/100", log_file)
    return score

def calculate_sentence_lengths_score(original_sent_lengths: List[int], paraphrased_sent_lengths: List[int], log_file: Optional[str] = None) -> float:
    """
    Calculate score based on individual sentence lengths comparison
    """
    # Handle edge cases
    if not original_sent_lengths:
        log_message("Warning: Original text has 0 sentences", log_file)
        return 0.0
        
    # If sentence counts differ, use the minimum length to compare
    min_len = min(len(original_sent_lengths), len(paraphrased_sent_lengths))
    
    if min_len == 0:
        return 0.0
    
    # Compare corresponding sentences up to the minimum length
    sentence_scores = []
    for i in range(min_len):
        if original_sent_lengths[i] == 0:
            continue  # Skip sentences with 0 words
        
        # 根据仿写句子长度与原句长度的关系分为两种情况
        if paraphrased_sent_lengths[i] <= original_sent_lengths[i]:
            # 情况1: 仿写句子长度少于或等于原句
            ratio = paraphrased_sent_lengths[i] / original_sent_lengths[i]
            sent_score = ratio * 100
            log_message(f"Sentence {i+1}: Original {original_sent_lengths[i]} words, Paraphrased {paraphrased_sent_lengths[i]} words", log_file)
            log_message(f"  Case: Shorter or equal - Ratio: {ratio:.2f}, Score: {sent_score:.2f}", log_file)
        else:
            # 情况2: 仿写句子长度多于原句
            extra_ratio = (paraphrased_sent_lengths[i] - original_sent_lengths[i]) / original_sent_lengths[i]
            # 限制惩罚上限
            penalty = extra_ratio * 100
            sent_score = 100 - penalty
            log_message(f"Sentence {i+1}: Original {original_sent_lengths[i]} words, Paraphrased {paraphrased_sent_lengths[i]} words", log_file)
            log_message(f"  Case: Longer - Extra ratio: {extra_ratio:.2f}, Penalty: {penalty:.2f}, Score: {sent_score:.2f}", log_file)
            
        sentence_scores.append(sent_score)
    
    # If we have no valid comparisons, return 0
    if not sentence_scores:
        return 0.0
        
    # 计算平均分数
    avg_score = sum(sentence_scores) / len(sentence_scores)
    
    log_message(f"Average sentence similarity score: {avg_score:.2f}/100", log_file)
    
    return avg_score

def evaluate_length(original_text: str, paraphrased_text: str, log_file: Optional[str] = None) -> float:
    # Input validation
    if not isinstance(original_text, str) or not isinstance(paraphrased_text, str):
        raise ValueError("Both original_text and paraphrased_text must be strings")
    
    if not original_text.strip():
        raise ValueError("Original text cannot be empty")
    
    nlp = spacy.load("en_core_web_sm")
    doc1 = nlp(original_text)
    doc2 = nlp(paraphrased_text)

    # Get sentences
    original_sents = list(doc1.sents)
    paraphrased_sents = list(doc2.sents)
    
    # Calculate sentence counts
    original_sent_count = len(original_sents)
    paraphrased_sent_count = len(paraphrased_sents)
    
    # Calculate word counts for each sentence
    original_sent_lengths = []
    for sent in original_sents:
        word_count = len([token for token in sent if not token.is_punct and not token.is_space])
        original_sent_lengths.append(word_count)
    
    paraphrased_sent_lengths = []
    for sent in paraphrased_sents:
        word_count = len([token for token in sent if not token.is_punct and not token.is_space])
        paraphrased_sent_lengths.append(word_count)
    
    # Total word counts (for logging)
    original_word_count = sum(original_sent_lengths)
    paraphrased_word_count = sum(paraphrased_sent_lengths)
    
    log_message("\n" + "="*50, log_file)
    log_message("Length Evaluation Details:", log_file)
    log_message("-"*50, log_file)
    log_message(f"Original text preview: {original_text[:100]}...", log_file)
    log_message(f"Original sentence count: {original_sent_count}", log_file)
    log_message(f"Original total word count: {original_word_count}", log_file)
    log_message(f"Original sentence lengths: {original_sent_lengths}", log_file)
    
    log_message(f"Paraphrased text preview: {paraphrased_text[:100]}...", log_file)
    log_message(f"Paraphrased sentence count: {paraphrased_sent_count}", log_file)
    log_message(f"Paraphrased total word count: {paraphrased_word_count}", log_file)
    log_message(f"Paraphrased sentence lengths: {paraphrased_sent_lengths}", log_file)
    
    # Calculate scores
    sent_count_score = calculate_sentence_score(original_sent_count, paraphrased_sent_count, log_file)
    sent_lengths_score = calculate_sentence_lengths_score(original_sent_lengths, paraphrased_sent_lengths, log_file)
    
    # 使用加权平均计算最终分数
    # final_score = (sent_count_score * 0.5) + (sent_lengths_score * 0.5)
    
    log_message(f"Sentence count score: {sent_count_score:.2f}/100 (60% weight)", log_file)
    log_message(f"Sentence lengths score: {sent_lengths_score:.2f}/100 (40% weight)", log_file)
    # log_message(f"Final length score: {final_score:.2f}/100", log_file)
    log_message("="*50 + "\n", log_file)
    
    return sent_count_score, sent_lengths_score


