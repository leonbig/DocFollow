# Using DocFollow

## Environment

From the repository root:

```bash
conda create -n docfollow python=3.12.9
conda activate docfollow
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The released corpus is in English. If using Chinese documents with the evaluation modules, also install `zh_core_web_sm` with `python -m spacy download zh_core_web_sm`.

## Model configuration

The main pipeline accepts separate generation and evaluation services through their model names, API keys, and base URLs. Set the following variables to your own service configuration (Bash syntax):

```bash
export MODEL_NAME="your-generation-model"
export OPENAI_API_KEY="your-generation-api-key"
export API_BASE_URL="your-generation-base-url"
export EVAL_MODEL_NAME="your-evaluation-model"
export EVAL_API_KEY="your-evaluation-api-key"
export EVAL_API_BASE_URL="your-evaluation-base-url"
```

## Run generation and evaluation

Run from `docfollow/`. Select one domain directory so that its immediate subdirectories correspond to writing scenarios:

```bash
cd docfollow
python final_main.py \
  --input_dir ../datasets/documents/education \
  --output_dir ../results \
  --theme_levels 1 \
  --levels 1 \
  --model_name "$MODEL_NAME" \
  --api_key "$OPENAI_API_KEY" \
  --base_url "$API_BASE_URL" \
  --eval_model "$EVAL_MODEL_NAME" \
  --eval_api_key "$EVAL_API_KEY" \
  --eval_base_url "$EVAL_API_BASE_URL"
```

Use `--theme_levels 1,2,3 --levels 1,2,3,4` to select all topic modification and task difficulty levels. Repeat with other domain directories as needed.

| Argument | Purpose |
| --- | --- |
| `--input_dir` | Domain directory containing scenario subdirectories with `.txt` source files |
| `--output_dir` | Root directory for generated documents, evaluation scores, and logs |
| `--theme_levels` | Comma-separated topic modification levels: R1, R2, R3 → `1,2,3` |
| `--levels` | Comma-separated cumulative difficulty levels: L1-L4 → `1,2,3,4` |
| `--model_name`, `--api_key`, `--base_url` | Generation service configuration |
| `--eval_model`, `--eval_api_key`, `--eval_base_url` | Evaluation service configuration |

Results are grouped by generation model and writing scenario under the selected output directory. The pipeline saves generated text, JSON evaluation scores, and execution logs.

## Prompting experiments

The prompting and temperature experiments use the same benchmark modules. From the repository root, make those modules available through `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/docfollow" python experiments/few_shot.py --help
```

Replace `few_shot.py` with `one_shot.py`, `cot.py`, `self_critic.py`, or `temperature.py` to inspect the corresponding arguments.

## DocFollower

[`experiments/docfollower/`](../experiments/docfollower/) contains the graph-guided generation framework. It reads `OPENAI_API_KEY` from the environment. Prepare its local input layout before running:

```text
experiments/docfollower/data/input/
├── original_text.txt
└── themes/
    ├── theme_level_1.txt
    ├── theme_level_2.txt
    └── theme_level_3.txt
```

Use the source document as `original_text.txt` and supply a target topic for each topic modification level. From the repository root:

```bash
cd experiments/docfollower
python main.py
```

The prompt templates are in `instruction.py`, and sentence splitting is implemented in `splitSentence.py`.
