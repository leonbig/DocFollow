# DocFollow and Evaluation Benchmark



### Project Description
This project implements a comprehensive benchmark for text generating and evaluation. It supports multiple Document-Following methods and evaluation metrics to assess the quality of generated texts across different difficulty levels and writing scenarios.

### Features
- Multiple different methods (4 different approaches)
- Comprehensive evaluation metrics:
  - Topic coherence
  - Length comparison
  - Sentence logic
  - Entity evaluation
  - Entity relation evaluation
- Support for different topic modification levels (1-3)
- Support for different text difficulties levels (1-4)
- Multiple experiment modes:
  - One-shot learning
  - Few-shot learning
  - Chain-of-Thought (COT)
  - Self-criticism
  - DocFollower
  - Temperature experiments

### Setup

1. Create a conda virtual environment with Python 3.12.9:
```bash
conda create -n xxx python=3.12.9
```

2. Activate the virtual environment:
```bash
conda activate xxx
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install required spaCy models:
```bash
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

### Usage

Run the main evaluation script:
```bash
python final_main.py --input_dir <input_directory> --output_dir <output_directory> --theme_levels <levels> --levels <levels> --model_name <model_name> --api_key <api_key> --base_url <base_url> --eval_model <eval_model> --eval_api_key <eval_api_key> --eval_base_url <eval_base_url>
```

Required parameters:
- `input_dir`: Directory containing input text files
- `output_dir`: Directory for saving evaluation results
- `theme_levels`: Comma-separated list of theme difficulty levels (1-3)
- `levels`: Comma-separated list of imitation difficulty levels (1-5)
- `model_name`: Name of the paraphrasing model to use
- `api_key`: OpenAI API key for imitation
- `base_url`: Base URL for imitation API requests
- `eval_model`: Name of the evaluation model to use
- `eval_api_key`: OpenAI API key for evaluation
- `eval_base_url`: Base URL for evaluation API requests

### Notes
- Evaluation results are saved in JSON format with detailed metrics
- Progress and timing information is logged during execution

