# DocFollow

Source code and source documents for the DocFollow text-generation and evaluation benchmark.

## Repository contents

- [`Source Code/`](Source%20Code/): generation methods, evaluators, experiment scripts, and dependencies.
- [`Source Data/`](Source%20Data/): 203 source documents in five domains.
- [`Source Code/readme.md`](Source%20Code/readme.md): the original software documentation, including command-line parameters.

The files under `Source Code/` and `Source Data/` are preserved from the supplementary software and data archives, including their original directory and file names.

### Code layout

```text
Source Code/
├── final_main.py             # Main generation and evaluation driver
├── requirements.txt          # Dependencies from the software archive
├── Evaluator/                # Topic, length, sentence, entity, and relation evaluation
├── Text_Generator/           # Text imitation and theme generation
├── Experiment/               # One-shot, few-shot, CoT, self-criticism, and temperature experiments
│   └── DocFollower/           # DocFollower implementation
└── tools/                    # Shared utilities and category definitions
```

### Source-document collection

| Domain directory | Documents |
| --- | ---: |
| `academic&engineer` | 54 |
| `education` | 39 |
| `finance&business` | 29 |
| `literature&art` | 40 |
| `politics&law` | 41 |
| **Total** | **203** |

Documents are plain-text files, grouped by domain and document type. Keep this hierarchy when using the original experiment scripts.

## Environment setup

The original software documentation specifies Python 3.12.9:

```bash
conda create -n docfollow python=3.12.9
conda activate docfollow
cd "Source Code"
pip install -r requirements.txt
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm
```

Generation and model-based evaluation require credentials for the chosen API services. See the [original software documentation](Source%20Code/readme.md) for the generation and evaluation arguments. The DocFollower implementation reads `OPENAI_API_KEY` from the environment.

## Reproduction status

This upload preserves the original software and data. All 24 Python files pass syntax parsing; full generation and evaluation have not been run.

The original `final_main.py` contains inconsistent relative imports and a misspelled module path (`TexText_Generatort_Rewriter`). These imports need to be corrected before the main script can run as described in the original documentation. The source files have been retained unchanged in this upload.

## License

The repository's existing [MIT license](LICENSE) is retained.
