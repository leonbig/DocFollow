import os
import datetime
import json
import glob
from pydoc import cli 
import argparse 
from Evaluator.length import evaluate_length
from Evaluator.sentence_rela_eval import SentenceRelaEvaluator
from Evaluator.theme_coherence import ThemeCoherenceEvaluator
from Evaluator.entity_eval import EntityEvaluator
from Evaluator.entity_relation_eval import EntityRelationEvaluator
from openai import OpenAI
from Text_Generator.theme_modify import generate_target_theme
from Text_Generator.get_original_theme import get_text_theme
from collections import defaultdict
import time
from functools import wraps
from tools.functions import retry_on_failure, save_scores_to_file, save_text_to_file

# PARAPHRASERS and EVALUATORS dictionaries remain the same
PARAPHRASERS = {
    "method1": lambda client, para_model, text, theme, temperature=1.0: text_imitation01(client, para_model, text, theme, temperature),
    "method2": lambda client, para_model, text, theme, temperature=1.0: text_imitation02(client, para_model, text, theme, temperature),
    "method3": lambda client, para_model, text, theme, temperature=1.0: text_imitation03(client, para_model, text, theme, temperature),
    "method4": lambda client, para_model, text, theme, temperature=1.0: text_imitation04(client, para_model, text, theme, temperature),
}

EVALUATORS = {
    "topic_coherence": lambda para, theme, client, eval_model, log_file: ThemeCoherenceEvaluator(client=client, eval_model=eval_model, log_file=log_file).evaluate_text(theme=theme, text=para),
    "length": lambda orig, para, log_file: evaluate_length(original_text=orig, paraphrased_text=para, log_file=log_file),
    "sentence_logic": lambda orig, para, client, eval_model, log_file: SentenceRelaEvaluator(client=client, eval_model=eval_model, log_file=log_file).evaluate_pair(original=orig, paraphrase=para),
    "entity_eval": lambda orig_sents, para_sents, client, eval_model, log_file: EntityEvaluator(client=client, eval_model=eval_model, log_file=log_file).evaluate_pair(orig_sents=orig_sents, para_sents=para_sents),
    "entity_relation_eval": lambda alignment, sents, entities, text_domain, client, eval_model, log_file: EntityRelationEvaluator(client=client, eval_model=eval_model, log_file=log_file).evaluate_pair(alignment=alignment, sents=sents, entities=entities, text_domain=text_domain),
}



def text_imitation01(client, para_model, original_text, new_theme, temperature=1.0):
    prompt_template = f"""
    # Role Definition
    You are an expert in text imitation.

    # Task Overview
    Analyze the original text and create a new text on a different topic while follow the following constrains.

    # Input
    Original Text:
    {original_text}

    Target topic:
    {new_theme}

    # Imitation Constrains

    1. Length Constrain:
       - The imitated text MUST contain exactly the same number of sentences as the original
       - Match the word count of each individual sentence as closely as possible to its corresponding sentence in the original
       - The total word count of the imitated text MUST match the original text exactly

    2. Topic Constrain:
       - The imitated content must focus entirely on the target topic
       - Ensure logical coherence and natural expression
       - Appropriately adjust technical terms or topic-specific vocabulary to fit the target topic

    # Output Requirements
        Output only the imitated text without explaining your analysis process or creative approach. Ensure the output text is complete, clearly formatted, and ready for immediate use.
"""

    messages = [{"role": "user", "content": prompt_template}]
    
    extra_params = {"temperature": temperature}
    response = client.chat.completions.create(
        model=para_model, 
        messages=messages,
        **extra_params
    )
    return response.choices[0].message.content.strip()


def text_imitation02(client, para_model, original_text, new_theme, temperature=1.0):
    prompt_template = f"""
    # Role Definition
    You are an expert in text imitation.

    # Task Overview
    Analyze the original text and create a new text on a different topic while follow the following constrains.

    # Input
    Original Text:
    {original_text}

    Target topic:
    {new_theme}

    # Imitation Constrains

    1. Length Constrain:
       - The imitated text MUST contain exactly the same number of sentences as the original
       - Match the word count of each individual sentence as closely as possible to its corresponding sentence in the original
       - The total word count of the imitated text MUST match the original text exactly

    2. Topic Constrain:
       - The imitated content must focus entirely on the target topic
       - Ensure logical coherence and natural expression
       - Appropriately adjust technical terms or topic-specific vocabulary to fit the target topic

    3. Sentence Relationship Constrain:
       - Each sentence in the imitated text MUST reference or be based on the corresponding sentence in the original text
       - The relationship between adjacent sentences in the imitated text MUST mirror the relationship between corresponding sentences in the original text
       - For example, if the third sentence in the original text provides an example for the second sentence, then the third sentence in the imitated text should also provide an example for the second sentence

    # Output Requirements
        Output only the imitated text without explaining your analysis process or creative approach. Ensure the output text is complete, clearly formatted, and ready for immediate use.
"""

    messages = [{"role": "user", "content": prompt_template}]
    
    extra_params = {"temperature": temperature}
    response = client.chat.completions.create(
        model=para_model, 
        messages=messages,
        **extra_params
    )
    return response.choices[0].message.content.strip()


def text_imitation03(client, para_model, original_text, new_theme, temperature=1.0):
    prompt_template = f"""
    # Role Definition
    You are an expert in text imitation.

    # Task Overview
    Analyze the original text and create a new text on a different topic while follow the following constrains.

    # Input
    Original Text:
    {original_text}

    Target topic:
    {new_theme}

    # Imitation Constrains

    1. Length Constrain:
       - The imitated text MUST contain exactly the same number of sentences as the original
       - Match the word count of each individual sentence as closely as possible to its corresponding sentence in the original
       - The total word count of the imitated text MUST match the original text exactly

    2. Topic Constrain:
       - The imitated content must focus entirely on the target topic
       - Ensure logical coherence and natural expression
       - Appropriately adjust technical terms or topic-specific vocabulary to fit the target topic

    3. Sentence Relationship Constrain:
       - Each sentence in the imitated text MUST reference or be based on the corresponding sentence in the original text
       - The relationship between adjacent sentences in the imitated text MUST mirror the relationship between corresponding sentences in the original text
       - For example, if the third sentence in the original text provides an example for the second sentence, then the third sentence in the imitated text should also provide an example for the second sentence

    4. Entity Type Constrain:
       - Entities in each sentence of the imitated text MUST correspond to entities in the matching sentence of the original text
       - Ensure that entity types remain consistent (e.g., if the original sentence mentions a person, a location, and an organization, the corresponding imitated sentence should also contain a person, a location, and an organization)
       - Maintain the same pattern and sequence of entities within each sentence when possible

    # Output Requirements
        Output only the imitated text without explaining your analysis process or creative approach. Ensure the output text is complete, clearly formatted, and ready for immediate use.
"""

    messages = [{"role": "user", "content": prompt_template}]
    
    extra_params = {"temperature": temperature}
    response = client.chat.completions.create(
        model=para_model, 
        messages=messages,
        **extra_params
    )
    return response.choices[0].message.content.strip()


def text_imitation04(client, para_model, original_text, new_theme, temperature=1.0):
    prompt_template = f"""
    # Role Definition
    You are an expert in text imitation.

    # Task Overview
    Analyze the original text and create a new text on a different topic while follow the following constrains.

    # Input
    Original Text:
    {original_text}

    Target topic:
    {new_theme}

    # Imitation Constrains

    1. Length Constrain:
       - The imitated text MUST contain exactly the same number of sentences as the original
       - Match the word count of each individual sentence as closely as possible to its corresponding sentence in the original
       - The total word count of the imitated text MUST match the original text exactly

    2. Topic Constrain:
       - The imitated content must focus entirely on the target topic
       - Ensure logical coherence and natural expression
       - Appropriately adjust technical terms or topic-specific vocabulary to fit the target topic

    3. Sentence Relationship Constrain:
       - Each sentence in the imitated text MUST reference or be based on the corresponding sentence in the original text
       - The relationship between adjacent sentences in the imitated text MUST mirror the relationship between corresponding sentences in the original text
       - For example, if the third sentence in the original text provides an example for the second sentence, then the third sentence in the imitated text should also provide an example for the second sentence

    4. Entity Type Constrain:
       - Entities in each sentence of the imitated text MUST correspond to entities in the matching sentence of the original text
       - Ensure that entity types remain consistent (e.g., if the original sentence mentions a person, a location, and an organization, the corresponding imitated sentence should also contain a person, a location, and an organization)
       - Maintain the same pattern and sequence of entities within each sentence when possible

    5. Entity Relationship Constrain:
       - The relationships between entities in each sentence of the imitated text MUST mirror the relationships between entities in the corresponding sentence of the original text
       - Ensure that relationship types remain consistent (e.g., if entity A is the parent of entity B in the original, then the corresponding entities in the imitated text should also have a parent-child relationship)
       - Preserve semantic roles between entities (e.g., if one entity is the agent and another is the recipient in the original, maintain this dynamic in the imitated text)

    # Output Requirements
        Output only the imitated text without explaining your analysis process or creative approach. Ensure the output text is complete, clearly formatted, and ready for immediate use.
"""

    messages = [{"role": "user", "content": prompt_template}]
    
    extra_params = {"temperature": temperature}
    response = client.chat.completions.create(
        model=para_model, 
        messages=messages,
        **extra_params
    )
    return response.choices[0].message.content.strip()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Text imitation and evaluation benchmark')
    parser.add_argument('--input_dir', type=str, 
                        help='Directory containing input folders with text files')
    parser.add_argument('--output_dir', type=str, 
                        help='Directory for output results')
    parser.add_argument('--theme_levels', type=str, 
                        help='Comma-separated list of theme difficulty levels to evaluate')
    parser.add_argument('--levels', type=str, 
                        help='Comma-separated list of text imitation difficulty levels to evaluate')
    parser.add_argument('--model_name', type=str, 
                        help='Name of the paraphrasing model to use')
    parser.add_argument('--api_key', type=str, 
                        help='OpenAI API key to use for imitation')
    parser.add_argument('--base_url', type=str,
                        help='Base URL for imitation API requests')
    parser.add_argument('--eval_model', type=str,
                        help='Name of the evaluation model to use')
    parser.add_argument('--eval_api_key', type=str,
                        help='OpenAI API key to use for evaluation')
    parser.add_argument('--eval_base_url', type=str,
                        help='Base URL for evaluation API requests')
    parser.add_argument('--temperatures', type=str, default='1.0',
                        help='Comma-separated list of temperature values to evaluate (e.g., "0.1,0.5,0.8,1.0")')

    args = parser.parse_args()
    
    # Parse theme levels and imitation levels from arguments
    theme_levels = [int(level) for level in args.theme_levels.split(',')]
    imitation_levels = [int(level) for level in args.levels.split(',')]
    model_name = args.model_name
    api_key = args.api_key
    base_url = args.base_url
    eval_model = args.eval_model
    eval_api_key = args.eval_api_key
    eval_base_url = args.eval_base_url
    temperatures = [float(temp) for temp in args.temperatures.split(',')]
    
    # Use parsed arguments
    input_dir = args.input_dir
    main_output_dir = args.output_dir
    
    print(f"Running with settings:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {main_output_dir}")
    print(f"  Theme levels: {theme_levels}")
    print(f"  Imitation levels: {imitation_levels}")
    print(f"  Using imitation model: {model_name}")
    print(f"  Imitation API key: [HIDDEN]")
    print(f"  Imitation Base URL: {base_url}")
    print(f"  Using evaluation model: {eval_model}")
    print(f"  Evaluation API key: [HIDDEN]")
    print(f"  Evaluation Base URL: {eval_base_url}")
    print(f"  Temperatures to evaluate: {temperatures}")

    # 设置模型有关的内容
    @retry_on_failure(max_retries=3)
    def initialize_clients():
        imitation_client_params = {
            "api_key": api_key,
            "base_url": base_url,
        }
            
        eval_client_params = {
            "api_key": eval_api_key,
            "base_url": eval_base_url,
        }
            
        imitation_client = OpenAI(**imitation_client_params)
        eval_client = OpenAI(**eval_client_params)
        return imitation_client, eval_client

    try:
        imitation_client, eval_client = initialize_clients()
    except Exception as e:
        print(f"Failed to initialize OpenAI Clients after multiple attempts: {e}")
        return 

    imitation_model = model_name
    print(f"Using paraphrasing model: {imitation_model}")
    print(f"Using evaluation model: {eval_model}")

    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input folder '{input_dir}' does not exist or is not a valid directory.")
        return

    # 创建主输出目录
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"All output will be saved under the '{main_output_dir}' folder.")

    # 初始化所有温度值的分数汇总
    all_temperatures_scores = {}
    
    # 对每个温度值进行实验
    for temperature in temperatures:
        print(f"\n{'='*20} Processing temperature: {temperature} {'='*20}")
        
        # 创建温度值对应的输出目录
        temp_dir_name = f"{model_name}_temp{temperature}"
        temp_output_dir = os.path.join(main_output_dir, temp_dir_name)
        os.makedirs(temp_output_dir, exist_ok=True)
        print(f"Output for temperature {temperature} will be saved under: {temp_output_dir}")

        # 初始化当前温度值的所有类别分数汇总
        all_categories_scores = {}
        
        # 查找所有类别文件夹（每个类别文件夹内有一个文本文件）
        category_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
        if not category_dirs:
            print(f"No category directories found in '{input_dir}'.")
            return
        
        print(f"Found {len(category_dirs)} category directories, starting processing...")
        
        # 处理每个类别文件夹
        for category_dir in category_dirs:
            category_name = os.path.basename(category_dir)
            print(f"\n{'='*15} Processing category: {category_name} {'='*15}")
            
            # 为当前类别创建输出目录
            category_output_dir = os.path.join(temp_output_dir, category_name)
            os.makedirs(category_output_dir, exist_ok=True)
            
            # 处理该类别文件夹下的文本
            category_scores = process_category(category_dir, category_output_dir, theme_levels, imitation_levels,
                                              imitation_client, eval_client, imitation_model, eval_model, temperature)
            
            # 记录该类别的分数
            if category_scores:
                all_categories_scores[category_name] = category_scores
        
        # 保存当前温度值的评估结果汇总
        temp_summary_file = os.path.join(temp_output_dir, f"{temp_dir_name}_summary.json")
        save_scores_to_file(all_categories_scores, temp_summary_file)
        print(f"\nTemperature {temperature} processed. Summary saved to: {temp_summary_file}")
        
        # 记录当前温度值的结果
        all_temperatures_scores[str(temperature)] = all_categories_scores
    
    # 保存所有温度值的评估结果汇总
    main_summary_file = os.path.join(main_output_dir, "all_temperatures_summary.json")
    save_scores_to_file(all_temperatures_scores, main_summary_file)
    print(f"\nAll temperatures processed. Overall summary saved to: {main_summary_file}")

    # 在所有处理完成后，计算并显示总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print("\n" + "="*50)
    print(f"总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
    print("="*50)

def process_category(category_dir, output_dir, theme_levels, imitation_levels, 
                    imitation_client, eval_client, imitation_model, eval_model, temperature=1.0):
    """处理单个类别目录，执行所有主题级别和模仿级别的评估"""
    
    print(f"Processing category directory: {category_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # 分数计算
    category_scores = defaultdict(lambda: defaultdict(dict))
    
    # 找到目录中的所有txt文件
    txt_files = glob.glob(os.path.join(category_dir, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in the category directory '{category_dir}'.")
        return None
    
    print(f"Found {len(txt_files)} .txt files.")
    
    # 只处理第一个文本文件（每个类别文件夹一个文本文件）
    if len(txt_files) > 1:
        print(f"Note: Found more than one text file. Will use the first one: {os.path.basename(txt_files[0])}")
    
    txt_file_path = txt_files[0]
    file_base_name = os.path.splitext(os.path.basename(txt_file_path))[0]
    
    # 读取原始文本
    try:
        with open(txt_file_path, "r", encoding="utf-8") as f:
            original_text = f.read().strip()
        if not original_text:
            print(f"  Warning: Empty file {file_base_name}.txt, skipping.")
            return None
        
        print(f"Successfully read text file. Text length: {len(original_text)} characters")
        
        # 保存原始文本到输出目录
        save_text_to_file(original_text, os.path.join(output_dir, "original.txt"))
    except Exception as e:
        print(f"Error reading {file_base_name}.txt: {e}, skipping.")
        return None
    
    # 遍历每个主题难度级别
    for theme_level in theme_levels:
        if theme_level < 1 or theme_level > 3:
            print(f"Invalid theme level {theme_level}, skipping. Valid range is 1-3.")
            continue
            
        print(f"\n{'='*15} Processing Theme Level: {theme_level} {'='*15}")
        
        # 创建主题级别目录
        theme_dir = os.path.join(output_dir, f"theme_level_{theme_level}")
        os.makedirs(theme_dir, exist_ok=True)
        
        # 获取或生成原始主题
        @retry_on_failure(max_retries=3)
        def get_theme(text, client, model):
            print(f"Attempting to get theme for text of length: {len(text)}")
            if not text or not text.strip():
                raise ValueError("Input text is empty or only contains whitespace")
            theme = get_text_theme(text, client, model)
            if theme is None or not theme.strip():
                raise ValueError("get_text_theme returned empty or None theme")
            return theme
        
        try:
            print("Getting original theme...")
            original_theme = get_theme(original_text, eval_client, eval_model)
            if not original_theme or not original_theme.strip():
                raise ValueError("Failed to get valid theme from text")
            print(f"Successfully got original theme: {original_theme}")
        except Exception as e:
            print(f"Error: Failed to get original topic after multiple attempts: {e}")
            print("Debug info:")
            print(f"  Text file path: {txt_file_path}")
            print(f"  Text length: {len(original_text) if original_text else 'None'}")
            print(f"  First 100 chars of text: {original_text[:100] if original_text else 'None'}")
            continue
        
        # 获取或生成目标主题
        themes_file = os.path.join(theme_dir, f"theme_level_{theme_level}_themes.json")
        target_theme = None
        
        # 尝试从现有文件加载主题
        if os.path.exists(themes_file):
            try:
                with open(themes_file, 'r', encoding='utf-8') as f:
                    themes_data = json.load(f)
                if "Target" in themes_data:
                    target_theme = themes_data["Target"]
                    print("  Successfully loaded target topic.")
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Error loading themes file: {e}, will regenerate.")
        
        # 如果没有成功加载，生成新主题
        if target_theme is None:
            print("  Generating new target theme...")
            try:
                @retry_on_failure(max_retries=3)
                def generate_theme(client, model, original_theme, text, level):
                    if not original_theme or not original_theme.strip():
                        raise ValueError("orig_theme must be a non-empty string")
                    if not text or not text.strip():
                        raise ValueError("text must be a non-empty string")
                    return generate_target_theme(client, model, original_theme, text, level)
                
                target_theme = generate_theme(eval_client, eval_model, original_theme, original_text, theme_level)
                themes_data = {
                    "Original": original_theme,
                    "Target": target_theme,
                    "Timestamp": datetime.datetime.now().isoformat()
                }
                save_text_to_file(json.dumps(themes_data, ensure_ascii=False, indent=2), themes_file)
                print("  New topic generated and saved.")
            except Exception as e:
                print(f"  Error: Failed to generate target topic after multiple attempts: {e}")
                print("  Debug info:")
                print(f"    Original theme: {original_theme}")
                print(f"    Theme level: {theme_level}")
                continue
        
        # 输出主题
        print(f"Original Theme: {original_theme}")
        print(f"Target Theme: {target_theme}")
        
        # 记录该主题级别的评估分数
        theme_level_scores = {}
        
        # 遍历每个仿写难度级别
        for level in imitation_levels:
            if level < 1 or level > 5:
                print(f"Invalid imitation level {level}, skipping. Valid range is 1-5.")
                continue
                
            print(f"\n  --- Level {level} imitate and evaluate ---")
            
            # 创建难度级别目录
            level_dir = os.path.join(theme_dir, f"level_{level}")
            os.makedirs(level_dir, exist_ok=True)
            
            # 定义评估日志文件和仿写文本文件路径
            eval_log_file = os.path.join(level_dir, f"level_{level}_evaluation.log")
            paraphrase_txt_file = os.path.join(level_dir, f"level_{level}_paraphrase.txt")
            
            # 执行仿写
            try:
                # 选择仿写方法
                method_name = ""
                if level == 1:
                    method_name = "method1"
                elif level == 2:
                    method_name = "method2"
                elif level == 3:
                    method_name = "method3"
                elif level == 4 or level == 5:
                    method_name = "method4"
                else:
                    print(f"  Error: Invalid level input '{level}', skipping.")
                    continue
                
                # 执行仿写
                @retry_on_failure(max_retries=3)
                def paraphrase_text(method, imitation_client, model, text, theme, temperature):
                    if method in PARAPHRASERS:
                        paraphraser = PARAPHRASERS[method]
                        return paraphraser(imitation_client, model, text, theme, temperature)
                    else:
                        raise ValueError(f"Invalid paraphrasing method: {method}")
                
                paraphrased_text = paraphrase_text(method_name, imitation_client, imitation_model, original_text, target_theme, temperature)
                
                if paraphrased_text:
                    print("  Text imitation completed, saving...")
                    save_text_to_file(paraphrased_text, paraphrase_txt_file)
                else:
                    print("  Error: Text imitation process did not return text. Skip this level.")
                    continue
                
            except Exception as e:
                print(f"Error: Failed to execute the imitation method '{method_name}' (Level {level}) after multiple attempts: {e}. Skip this level.")
                continue
            
            # 创建评估日志文件头部
            try:
                with open(eval_log_file, 'w', encoding='utf-8') as log_f:
                    log_f.write(f"Evaluation Log for: {file_base_name}.txt\n")
                    log_f.write(f"Input Directory: {category_dir}\n")
                    log_f.write(f"Topic Level: {theme_level}, Evaluation Level: {level}\n")
                    log_f.write(f"Original Theme: {original_theme}\n")
                    log_f.write(f"Target Theme: {target_theme}\n")
                    log_f.write(f"Model: {imitation_model}\n")
                    log_f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    log_f.write(f"{'='*20}\n\n")
            except Exception as e:
                print(f"  Warning: Unable to create evaluation log header: {e}")
            
            # 初始化该级别的分数
            file_scores = {}
            all_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            
            # 执行评估
            @retry_on_failure(max_retries=3)
            def execute_evaluation(evaluator_name, *args):
                return EVALUATORS[evaluator_name](*args)
            
            try:
                alignment, sents, entities = None, None, None
                
                # 根据难度级别执行不同的评估
                if level == 1:
                    print("==========Evaluation: topic_coherence==========")
                    topic_coherence_score = execute_evaluation("topic_coherence", paraphrased_text, target_theme, eval_client, eval_model, eval_log_file)
                    file_scores["topic_coherence"] = topic_coherence_score
                    all_scores[theme_level][level]["topic_coherence"].append(topic_coherence_score)
                    
                    print("==========Evaluation: length=========")
                    sent_count_score, sent_lengths_score = execute_evaluation("length", original_text, paraphrased_text, eval_log_file)
                    file_scores["length"] = {"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score}
                    all_scores[theme_level][level]["length"].append({"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score})

                elif level == 2:
                    print("==========Evaluation: topic_coherence==========")
                    topic_coherence_score = execute_evaluation("topic_coherence", paraphrased_text, target_theme,  eval_client, eval_model, eval_log_file)
                    file_scores["topic_coherence"] = topic_coherence_score
                    all_scores[theme_level][level]["topic_coherence"].append(topic_coherence_score)
                    
                    print("==========Evaluation: length==========")
                    sent_count_score, sent_lengths_score = execute_evaluation("length", original_text, paraphrased_text, eval_log_file)
                    file_scores["length"] = {"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score}
                    all_scores[theme_level][level]["length"].append({"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score})
                    
                    print("==========Evaluation: sentence_logic==========")
                    sentence_logic_score, original_sentences, paraphrased_sentences, alignment = execute_evaluation("sentence_logic", original_text, paraphrased_text, eval_client, eval_model, eval_log_file)
                    file_scores["sentence_logic"] = sentence_logic_score
                    all_scores[theme_level][level]["sentence_logic"].append(sentence_logic_score)

                elif level == 3:
                    print("==========Evaluation: topic_coherence==========")
                    topic_coherence_score = execute_evaluation("topic_coherence", paraphrased_text, target_theme, eval_client, eval_model, eval_log_file)
                    file_scores["topic_coherence"] = topic_coherence_score
                    all_scores[theme_level][level]["topic_coherence"].append(topic_coherence_score)
                    
                    print("==========Evaluation: length==========")
                    sent_count_score, sent_lengths_score = execute_evaluation("length", original_text, paraphrased_text, eval_log_file)
                    file_scores["length"] = {"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score}
                    all_scores[theme_level][level]["length"].append({"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score})
                    
                    print("==========Evaluation: sentence_logic==========")
                    sentence_logic_score, original_sentences, paraphrased_sentences, alignment = execute_evaluation("sentence_logic", original_text, paraphrased_text, eval_client, eval_model, eval_log_file)
                    file_scores["sentence_logic"] = sentence_logic_score
                    all_scores[theme_level][level]["sentence_logic"].append(sentence_logic_score)
                    
                    print("==========Evaluation: entity_eval==========")
                    entity_score, alignment, sents, entities, text_domain = execute_evaluation("entity_eval", original_sentences, paraphrased_sentences, eval_client, eval_model, eval_log_file)
                    file_scores["entity_eval"] = entity_score
                    all_scores[theme_level][level]["entity_eval"].append(entity_score)

                elif level == 4:
                    print("==========Evaluation: topic_coherence==========")
                    topic_coherence_score = execute_evaluation("topic_coherence", paraphrased_text, target_theme, eval_client, eval_model, eval_log_file)
                    file_scores["topic_coherence"] = topic_coherence_score
                    all_scores[theme_level][level]["topic_coherence"].append(topic_coherence_score)
                    
                    print("==========Evaluation: length==========")
                    sent_count_score, sent_lengths_score = execute_evaluation("length", original_text, paraphrased_text, eval_log_file)
                    file_scores["length"] = {"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score}
                    all_scores[theme_level][level]["length"].append({"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score})
                    
                    print("==========Evaluation: sentence_logic==========")
                    sentence_logic_score, original_sentences, paraphrased_sentences, alignment = execute_evaluation("sentence_logic", original_text, paraphrased_text, eval_client, eval_model, eval_log_file)
                    file_scores["sentence_logic"] = sentence_logic_score
                    all_scores[theme_level][level]["sentence_logic"].append(sentence_logic_score)
                    
                    print("==========Evaluation: entity_eval==========")
                    entity_score, alignment, sents, entities, text_domain = execute_evaluation("entity_eval", original_sentences, paraphrased_sentences, eval_client, eval_model, eval_log_file)
                    file_scores["entity_eval"] = entity_score
                    all_scores[theme_level][level]["entity_eval"].append(entity_score)
                    
                    print("==========Evaluation: entity_relation_eval==========")
                    if alignment is not None and sents is not None and entities is not None:
                        entity_relation_score = execute_evaluation("entity_relation_eval", alignment, sents, entities, text_domain, eval_client, eval_model, eval_log_file)
                        file_scores["entity_relation_eval"] = entity_relation_score
                        all_scores[theme_level][level]["entity_relation_eval"].append(entity_relation_score)
                    else:
                        print("    Warning: Unable to get necessary inputs from entity_eval, skip entity_relation_eval.")
                        with open(eval_log_file, 'a', encoding='utf-8') as log_f:
                            log_f.write("\nWARNING: Skipped entity_relation_eval due to missing inputs from entity_eval.\n")

                elif level == 5:
                    print("==========Evaluation: topic_coherence==========")
                    topic_coherence_score = execute_evaluation("topic_coherence", paraphrased_text, target_theme, eval_client, eval_model, eval_log_file)
                    file_scores["topic_coherence"] = topic_coherence_score
                    all_scores[theme_level][level]["topic_coherence"].append(topic_coherence_score)
                    
                    print("==========Evaluation: length==========")
                    sent_count_score, sent_lengths_score = execute_evaluation("length", original_text, paraphrased_text, eval_log_file)
                    file_scores["length"] = {"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score}
                    all_scores[theme_level][level]["length"].append({"sent_count_score": sent_count_score, "sent_lengths_score": sent_lengths_score})
                    
                    print("==========Evaluation: sentence_logic==========")
                    sentence_logic_score, original_sentences, paraphrased_sentences, alignment = execute_evaluation("sentence_logic", original_text, paraphrased_text, eval_client, eval_model, eval_log_file)
                    file_scores["sentence_logic"] = sentence_logic_score
                    all_scores[theme_level][level]["sentence_logic"].append(sentence_logic_score)
                    
                    print("==========Evaluation: amr_similarity==========")
                    amr_similarity_score = execute_evaluation("amr_similarity", original_text, paraphrased_text, eval_client, eval_model, eval_log_file)
                    file_scores["amr_similarity"] = amr_similarity_score
                    all_scores[theme_level][level]["amr_similarity"].append(amr_similarity_score)
                
                # 保存该级别的分数
                scores_file = os.path.join(level_dir, f"level_{level}_scores.json")
                save_scores_to_file(file_scores, scores_file)
                print(f"  Level {level} evaluation completed. Scores saved to: {scores_file}")
                
                # 记录该级别的分数到主题级别结果中
                theme_level_scores[f"level_{level}"] = file_scores
                
            except Exception as e:
                print(f"  Error: During evaluation step (Level {level}) an error occurred after multiple attempts: {e}")
                try:
                    with open(eval_log_file, 'a', encoding='utf-8') as log_f:
                        log_f.write(f"\nERROR during evaluation step: {e}\n")
                except Exception as log_e:
                    print(f"  And unable to write error to log file: {log_e}")
        
        # 保存该主题级别的所有评估结果
        theme_summary_file = os.path.join(theme_dir, f"theme_level_{theme_level}_summary.json")
        save_scores_to_file(theme_level_scores, theme_summary_file)
        print(f"\nTheme level {theme_level} completed. Summary saved to: {theme_summary_file}")
        
        # 记录该主题级别的分数到类别结果中
        category_scores[f"theme_level_{theme_level}"] = theme_level_scores
    
    # 保存该类别的所有评估结果汇总
    category_summary_file = os.path.join(output_dir, f"{os.path.basename(output_dir)}_summary.json")
    save_scores_to_file(category_scores, category_summary_file)
    print(f"\nCategory processing completed. Summary saved to: {category_summary_file}")
    
    return category_scores

if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    
    main()