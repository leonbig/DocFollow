import os
import spacy
from typing import List, Optional, Dict
import time
import json

def spacy_sentence_split(text: str, model: str = "en_core_web_sm") -> List[str]:

    try:
        nlp = spacy.load(model)
    except OSError:
        print(f"需要下载 spaCy 模型: {model}")
        print(f"请运行: python -m spacy download {model}")
        raise
    
    # 清除已有的句子分割器（如果存在）
    if "sentencizer" in nlp.pipe_names:
        nlp.remove_pipe("sentencizer")
    
    # 添加句子分割器组件
    nlp.add_pipe("sentencizer")
    
    # 可以选择性禁用其他不需要的组件以提高速度
    # 但保留 tokenizer 和新添加的 sentencizer
    for pipe in nlp.pipe_names:
        if pipe not in ["tokenizer", "sentencizer"]:
            nlp.disable_pipe(pipe)
    
    # 处理文本
    doc = nlp(text)
    
    # 提取句子
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return sentences

def split_text_to_json(text: str, model: str = "en_core_web_sm") -> Dict:

    # 使用spacy进行句子分割
    sentences = spacy_sentence_split(text, model)
    
    # 构建输出格式
    output = {
        "sentence_count": len(sentences),
        "sentences": [
            {
                "index": i + 1,
                "content": sent
            }
            for i, sent in enumerate(sentences)
        ]
    }
    
    return output

def sentence_split(text: str) -> str:

    result = split_text_to_json(text)
    
    # Return the result as a JSON string
    return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    # Example for when running the script directly
    sample_text = """
    Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. 
    It enables computers to understand, interpret, and generate human language in a valuable way. 
    The main tasks in NLP include: text classification, sentiment analysis, machine translation, and question answering. 
    Modern NLP systems are based on machine learning algorithms, particularly deep learning models like BERT and GPT. 
    These models have achieved remarkable success in various NLP tasks.
    
    SpaCy is a popular open-source library for NLP in Python. 
    It provides pre-trained models for multiple languages and offers efficient implementations of common NLP tasks. 
    The library is known for its speed and ease of use. 
    It's particularly good at tokenization, part-of-speech tagging, and named entity recognition.
    """
    
    print("输入文本示例:")
    print("-" * 50)
    print(sample_text)
    print("-" * 50)
    
    print("\n处理结果:")
    print("-" * 50)
    result = sentence_split(sample_text)
    print(result)