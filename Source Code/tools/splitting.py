import os
import spacy
from typing import List, Optional
import time

def spacy_sentence_split(text: str, model: str = "en_core_web_sm") -> List[str]:
    try:
        nlp = spacy.load(model)
    except OSError:
        print(f"需要下载 spaCy 模型: {model}")
        print(f"请运行: python -m spacy download {model}")
        raise
    
    if "sentencizer" in nlp.pipe_names:
        nlp.remove_pipe("sentencizer")
    
    nlp.add_pipe("sentencizer")
    
    for pipe in nlp.pipe_names:
        if pipe not in ["tokenizer", "sentencizer"]:
            nlp.disable_pipe(pipe)
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

if __name__ == "__main__":
    print("\n支持的语言模型:")
    print("- 中文: zh_core_web_sm")
    print("- 德文: de_core_news_sm")
    print("- 法文: fr_core_news_sm")
    print("- 西班牙文: es_core_news_sm")
    print("可以使用 'python -m spacy download 模型名' 安装其他语言模型")
