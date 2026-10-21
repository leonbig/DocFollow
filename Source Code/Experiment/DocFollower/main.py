import os
import json
import random
from pydantic import BaseModel
import json
from instruction import PROMPTS
from openai import OpenAI
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import multiprocessing
from multiprocessing import Pool, Lock
import os.path
import argparse
import sys
from splitSentence import sentence_split

if 'OPENAI_API_KEY' not in os.environ:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set your OpenAI API key as an environment variable before running this script.")
    print("Example: set OPENAI_API_KEY=your_api_key_here (on Windows)")
    print("         export OPENAI_API_KEY=your_api_key_here (on Linux/Mac)")
    sys.exit(1)

client = OpenAI()
openai_client = OpenAI()
dir_lock = Lock()

def get_next_file_number(base_name, i):
    while True:
        file_name = f"{base_name}{i:03d}.json"
        if not os.path.exists(file_name):
            return i
        i += 1

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=100))
def run_output(step_num, system_prompt, prompt_key, **kwargs):
    try:
        input_prompt = PROMPTS[prompt_key].format(**kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ]
        response = client.chat.completions.create()
        output = response.choices[0].message.content
        print(f"Step {step_num} output:\n", output)
        return output
    except Exception as e:
        print(f"Error in run_json step {step_num}: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_json(step_num, system_prompt, prompt_key, **kwargs):
    try:
        input_prompt = PROMPTS[prompt_key].format(**kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ]
        response = client.chat.completions.create()
        output = response.choices[0].message.content
        print(f"Step {step_num} output:\n")
        return output
    except Exception as e:
        print(f"Error in run_json step {step_num}: {str(e)}")
        raise

def run_gpt(step_num, system_prompt, prompt_key, **kwargs):
    try:
        input_prompt = PROMPTS[prompt_key].format(**kwargs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt}
        ]
        response = client.chat.completions.create()
        output = response.choices[0].message.content
        print(f"Step {step_num} output:\n", output)
        return output
    except Exception as e:
        print(f"Error in run_json step {step_num}: {str(e)}")
        raise

def check_end(step_num, prompt_key, **kwargs):
    prompt = PROMPTS[prompt_key].format(**kwargs)
    chat_completion = client.chat.completions.create()
    output = chat_completion.choices[0].message.content
    print(f"Step {step_num} output:\n", output)
    return output

def add_index(json_str, index):
    data = json.loads(json_str)
    new_data = {
        "index": index,
        **data
    }
    return json.dumps(new_data, ensure_ascii=False, indent=2)

def save_entities_result(new_result, output_file):
    if isinstance(new_result, str):
        new_result = json.loads(new_result)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    all_results.append(new_result)

    with dir_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

def save_results(results_dir, file_name="fintech_results_1.json", **kwargs):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    file_path = os.path.join(results_dir, file_name)

    def encode_dict(d):
        if isinstance(d, dict):
            return {k: encode_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [encode_dict(i) for i in d]
        elif isinstance(d, str):
            return d.encode('utf-8').decode('utf-8')
        return d

    encoded_data = encode_dict(kwargs)

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        except Exception as e:
            print(f"Error reading existing file: {e}")
            existing_data = []
    else:
        existing_data = []

    existing_data.append(encoded_data)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

n=1

def graphAnalyse_text(original_text, n, i):
    global results_dir
    
    print(original_text)
    original_text = sentence_split(original_text)
    original_text = json.loads(original_text)
    current_sentence = ""
    already_output = ""
    
    for sentence in original_text["sentences"]:
        current_sentence = sentence["content"]
        index = sentence["index"]
        extracted_entities = run_json(2, system_json_0, "prompt_EntityExtract", current_sentence=current_sentence, original_text=original_text)
        extracted_entities = add_index(extracted_entities, index)
        extracted_relations = run_json(3, system_json_0, "prompt_RelationExtract", current_sentence=current_sentence, original_text=original_text, extracted_entities=extracted_entities)
        extracted_relations = add_index(extracted_relations, index)

        number = n
        paper_dir = os.path.join(results_dir, f"paper_{i:03d}")
        os.makedirs(paper_dir, exist_ok=True)
        
        save_entities_result(extracted_entities, output_file=os.path.join(paper_dir, f"entities_results{n:02d}.json"))
        save_entities_result(extracted_relations, output_file=os.path.join(paper_dir, f"relations_results{n:02d}.json"))
        print("当前处理中", current_sentence)
        print("当前总输出", already_output)

for n in range(1, 200):
    exec(f'Decrease_Entity_{n} = " "')
    exec(f'all_entities_{n} = " "')
    exec(f'summary_{n} = " "')

summary = ''
n=1

def process_text(original_text, index):
    global results_dir
    
    strings = []
    check = ''
    Decrease_Entity = ''
    n=0
    while "END" not in check:
        if n== 0:
            graphAnalyse_text(original_text, n, index)
            paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
            all_entities = read_json_file(os.path.join(paper_dir, f"entities_results{n:02d}.json"))
        n = n+1
        print(all_entities)
        Decrease_Entity= run_json(1, system_json_1, "prompt_EntityDecrease", original_text=original_text, extracted_entities=all_entities)
        exec(f'Decrease_Entity_{n} = Decrease_Entity')
        all_entities = run_json(1, system_json_1, "prompt_LeftEntity", extracted_entities=all_entities, Decrease_Entity=Decrease_Entity)
        exec(f'all_entities_{n} = all_entities')
        summary = run_json(1, system_json_1, "prompt_summary", original_text=original_text, all_entities=all_entities, Decrease_Entity=Decrease_Entity)

        Decrease_Entity_Relation = run_json(1, system_json_1, "prompt_HighRelationExtract", original_text=original_text, extracted_entities=Decrease_Entity)
        exec(f'summary_{n} = summary')
        exec(f'print(summary_{n})')

        strings.append(summary)
        original_text = summary
        check = run_json(1, system_json_1, "prompt_check", all_entities=all_entities)
        print(n,"现在是n次")
        number = n
        
        paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
        
        save_entities_result(all_entities, output_file=os.path.join(paper_dir, f"entities_results{number:02d}.json"))
        save_entities_result(Decrease_Entity, output_file=os.path.join(paper_dir, f"Decrease_Entity{number:02d}.json"))
        save_entities_result(Decrease_Entity_Relation, output_file=os.path.join(paper_dir, f"Decrease_Entity_Relation{number:02d}.json"))

    Decrease_Entity_Relation_max = run_json(1, system_json_1, "prompt_HighRelationExtract", original_text=original_text, extracted_entities=all_entities)
    Decrease_Entity_max = all_entities
    summary_max = summary
    
    paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
    
    save_entities_result(Decrease_Entity_Relation_max, output_file=os.path.join(paper_dir, "Decrease_Entity_Relation_max.json"))
    save_entities_result(Decrease_Entity_max, output_file=os.path.join(paper_dir, "Decrease_Entity_max.json"))
    return n, strings

def read_text_and_topic_files(content_file_path, topic_file_path):
    try:
        if not os.path.exists(content_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alternative_path = os.path.join(script_dir, os.path.basename(content_file_path))
            
            if os.path.exists(alternative_path):
                content_file_path = alternative_path
            else:
                working_dir = os.getcwd()
                alternative_path = os.path.join(working_dir, os.path.basename(content_file_path))
                if os.path.exists(alternative_path):
                    content_file_path = alternative_path
        
        print(f"Attempting to open content file at: {content_file_path}")
        with open(content_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if not os.path.exists(topic_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alternative_path = os.path.join(script_dir, os.path.basename(topic_file_path))
            
            if os.path.exists(alternative_path):
                topic_file_path = alternative_path
            else:
                working_dir = os.getcwd()
                alternative_path = os.path.join(working_dir, os.path.basename(topic_file_path))
                if os.path.exists(alternative_path):
                    topic_file_path = alternative_path
        
        print(f"Attempting to open topic file at: {topic_file_path}")
        with open(topic_file_path, 'r', encoding='utf-8') as file:
            topic = file.read().strip()
        
        topic = topic.replace("'s", "").replace("'", "").replace('"', "")
        
        if len(topic.split()) > 5:
            topic_words = topic.split()[:5]
            topic = " ".join(topic_words)
        
        return content, topic
    except Exception as e:
        print(f"Error reading files: {str(e)}")
        return None, None

def process_single_file(content_file_path, topic_file_path, results_dir, index=1, txt_output_dir=None):
    log_file = os.path.join(results_dir, "processing_log.txt")
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*50}\n")
        log.write(f"开始处理文件: {content_file_path} (内容), {topic_file_path} (主题)\n")
        log.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"{'='*50}\n\n")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"处理文件: {content_file_path} (内容), {topic_file_path} (主题)")
    print(f"{'='*50}\n")
    
    original_text, target_topic = read_text_and_topic_files(content_file_path, topic_file_path)
    
    if original_text and target_topic:
        print(f"内容文件: {content_file_path}")
        print(f"主题文件: {topic_file_path}")
        print(f"主题: {target_topic}")
        print(f"结果将保存到: {results_dir}/paper_{index:03d}")
        
        paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
        os.makedirs(paper_dir, exist_ok=True)
        
        try:
            generate_text(original_text, target_topic, index, results_dir, txt_output_dir)
            print(f"文件处理完成")
            
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"[{time.strftime('%H:%M:%S')}] 成功处理文件\n")
                log.write(f"主题: {target_topic}\n")
                log.write(f"输出目录: paper_{index:03d}\n\n")
            
            return True
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"[{time.strftime('%H:%M:%S')}] 处理文件失败\n")
                log.write(f"错误: {str(e)}\n\n")
            
            return False
    else:
        print(f"无法读取文件内容或主题")
        
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f"[{time.strftime('%H:%M:%S')}] 无法读取文件内容或主题\n\n")
        
        return False

def generate_text(paper_data, target_topic, index, results_dir, txt_output_dir=None):
    k,summary_list = process_text(paper_data, index=index)
    already_output = ""
    original_text_0 = paper_data
    max = k
    while k != 1:
        if k== max:
            paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
            Decrease_Entity_Relation = read_json_file(os.path.join(paper_dir, "Decrease_Entity_Relation_max.json"))
            Decrease_Entity = read_json_file(os.path.join(paper_dir, "Decrease_Entity_max.json"))
            original_text = summary_list[k-1]
            Higher_Entity = run_json(1, system_json_1, "prompt_HighEntityExtract", original_text=original_text, target_topic=target_topic, Decrease_Entity=Decrease_Entity, Decrease_Relation=Decrease_Entity_Relation, target_text_draft=already_output)

            generation_text = run_json(1, system_json_3, "prompt_OutputWriting", original_text=original_text, target_topic=target_topic, already_output=already_output, entity_analysis=Higher_Entity)
            already_output = generation_text

        number = k
        paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
        Decrease_Entity = read_json_file(os.path.join(paper_dir, f"Decrease_Entity{number:02d}.json"))
        Decrease_Entity_Relation = read_json_file(os.path.join(paper_dir, f"Decrease_Entity_Relation{number:02d}.json"))
        original_text = summary_list[k-2]

        Higher_Entity = run_json(1, system_json_1, "prompt_HighEntityExtract", original_text=original_text, target_topic=target_topic, target_text_draft=already_output, Decrease_Entity=Decrease_Entity, Decrease_Relation=Decrease_Entity_Relation)
        save_entities_result(Higher_Entity, output_file=os.path.join(paper_dir, f"Higher_Entity{number:02d}.json"))
        generation_text = run_json(1, system_json_3, "prompt_OutputWriting", original_text=original_text, target_topic=target_topic, already_output=already_output, entity_analysis=Higher_Entity)
        already_output = generation_text

        k = k-1
    number = 1
    paper_dir = os.path.join(results_dir, f"paper_{index:03d}")
    Decrease_Entity = read_json_file(os.path.join(paper_dir, f"Decrease_Entity{number:02d}.json"))
    Decrease_Entity_Relation = read_json_file(os.path.join(paper_dir, f"Decrease_Entity_Relation{number:02d}.json"))
    original_text = original_text_0
    Higher_Entity = run_json(1, system_json_1, "prompt_HighEntityExtract", original_text=original_text, target_topic=target_topic, target_text_draft=already_output, Decrease_Entity=Decrease_Entity, Decrease_Relation=Decrease_Entity_Relation)
    generation_text = run_output(1, system_json_2, "prompt_OutputWriting", original_text=original_text, target_topic=target_topic, already_output=already_output, entity_analysis=Higher_Entity)
    already_output = generation_text
    already_output = run_output(1, system_json_2, "prompt_RefineWriting", original_text=original_text, target_topic=target_topic, generated_text=generation_text)
    
    save_results(results_dir, file_name=f"{target_topic}_results.json", original_text=original_text_0, OURS=already_output, before_refined=generation_text, topic=target_topic)
    
    if txt_output_dir is None:
        txt_output_dir = os.path.join(results_dir, f"{target_topic}_txt_outputs")
    os.makedirs(txt_output_dir, exist_ok=True)
    
    final_output_path = os.path.join(txt_output_dir, f"{target_topic}_final_output.txt")
    with open(final_output_path, 'w', encoding='utf-8') as f:
        f.write(already_output)
    print(f"最终输出已保存到: {final_output_path}")
    
    before_refined_path = os.path.join(txt_output_dir, f"{target_topic}_before_refined.txt")
    with open(before_refined_path, 'w', encoding='utf-8') as f:
        f.write(generation_text)
    print(f"优化前文本已保存到: {before_refined_path}")

def read_text_file(file_path):
    try:
        if not os.path.exists(file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alternative_path = os.path.join(script_dir, os.path.basename(file_path))
            
            if os.path.exists(alternative_path):
                file_path = alternative_path
            else:
                working_dir = os.getcwd()
                alternative_path = os.path.join(working_dir, os.path.basename(file_path))
                if os.path.exists(alternative_path):
                    file_path = alternative_path
        
        print(f"Attempting to open file at: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        filename = os.path.basename(file_path)
        topic = os.path.splitext(filename)[0]
        
        topic = topic.replace("'s", "").replace("'", "").replace('"', "")
        
        if len(topic.split()) > 5:
            topic_words = topic.split()[:5]
            topic = " ".join(topic_words)
        
        return content, topic
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None, None

if __name__ == "__main__":
    content_file = "data/input/original_text.txt"
    content_dir = os.path.dirname(content_file)
    imitated_theme_dir = "data/input/themes"
    theme_files = [
        os.path.join(imitated_theme_dir, "theme_level_1.txt"),
        os.path.join(imitated_theme_dir, "theme_level_2.txt"),
        os.path.join(imitated_theme_dir, "theme_level_3.txt")
    ]
    results_dir = "data/output"
    os.makedirs(results_dir, exist_ok=True)
    txt_output_dir = os.path.join(content_dir, "results")
    os.makedirs(txt_output_dir, exist_ok=True)
    start_time = time.time()
    print(f"Starting to process {len(theme_files)} theme files")
    
    for i, topic_file in enumerate(theme_files, 1):
        print(f"\n{'='*50}")
        print(f"Processing theme file {i}/{len(theme_files)}: {topic_file}")
        print(f"{'='*50}\n")
        
        level_output_dir = os.path.join(txt_output_dir, f"level_{i}")
        os.makedirs(level_output_dir, exist_ok=True)
        
        try:
            success = process_single_file(content_file, topic_file, results_dir, i, level_output_dir)
            print(f"Processing result for level {i}: {'Success' if success else 'Failed'}")
        except Exception as e:
            print(f"Error processing theme level {i}: {str(e)}")
            log_file = os.path.join(results_dir, "processing_log.txt")
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(f"\n[{time.strftime('%H:%M:%S')}] Error processing theme level {i}: {str(e)}\n")
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal processing time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    print("All theme levels processed.")

system_json_0 = """
You are a professional text analysis assistant whose main task is to perform sentence identification and analysis of text. You need to:

1. Accurately identify English sentence boundaries, mainly using periods(.) as delimiters.
2. Accurately handle special characters in the text including numbers, punctuation marks, quotation marks, etc.  
3. Output results strictly according to the required JSON format
4. Only output JSON format results, without adding any additional explanations or notes
5. Ensure the output JSON format is standard and can be correctly parsed by programs

Notes:
- Maintain sentence completeness, including all text and punctuation marks
- Ensure sequence numbers start from 1 and increment
- Ignore empty lines and lines containing only whitespace characters 
- Maintain original order when processing text
- Do not make any modifications or optimizations to sentence content
- Do not add any comments or explanations

You should always respond in JSON format, even in exceptional cases. Maintaining consistency in output is important for subsequent program processing.
"""

system_json_1 = """
You are a professional text analysis assistant whose main task is to perform sentence identification and analysis of text. You need to:

1. Output results strictly according to the required JSON format
2. Only output JSON format results, without adding any additional explanations or notes

You should always respond in JSON format, even in exceptional cases. Maintaining consistency in output is important for subsequent program processing.
"""

system_json_2 = """
You are a text mimicry specialist. Produce new text that matches reference text patterns.

RULES:
1. Match structure & style exactly with Style reference text (A)
2. No blind copying words from Style reference text (A)
3. No blind copying entity names/terms from Style reference text (A)
4. Pay special attention to mimicking Style reference text (A) in terms of logical words such as conjunctions and prepositions.
5. Replace all entities with topic-appropriate equivalents
6. Keep same:
  - Paragraph length ratios
  - Argument flow
  - Professional depth
  - Entity relationship patterns

PROCESS:
1. Analyze reference text
2. Replace entities 
3. Generate parallel content
4. Verify no copied words/terms
5. Output clean text

OUTPUT:
Requirement: Output complete text directly, without any additional content.
"""

system_json_3 = """
You are a text mimicry specialist. Produce new text that matches reference text patterns.

RULES:
1. Match structure & style exactly with Style reference text (A)
2. No blind copying words from Style reference text (A)
3. No blind copying entity names/terms from Style reference text (A)
4. Pay special attention to mimicking Style reference text (A) in terms of logical words such as conjunctions and prepositions.
5. Replace all entities with topic-appropriate equivalents
6. Keep same:
  - Paragraph length ratios
  - Entity relationship patterns

PROCESS:
1. Analyze reference text
2. Replace entities 
3. Generate parallel content
4. Verify no copied words/terms
5. Output clean text

OUTPUT:
You should always respond in JSON format, even in exceptional cases. Maintaining consistency in output is important for subsequent program processing.
"""