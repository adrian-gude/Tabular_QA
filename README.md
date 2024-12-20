# Tabular_QA

Question Answering on Tabular Data SemEval 2025 - Task 8

## Installation

Requires:

- Cuda

```bash
pip install -r requirements.txt
```

### File Structure

```bash
.
├── LICENSE
├── logs
├── main.py   # Program entrypoint, tabular QA
├── README.md
├── requirements.txt
└── src
    ├── code_fixer.py       # Fixed code error from generated code
    └── column_selector.py  # Get columns related to the question
```

## Usage

```bash
usage: Tabular-QA [-h] [-m [MODEL]] [-t [{task-1,task-2,all}]] [-b [BATCH_SIZE]] [-n [NUMBER_ROWS]] [-z] [--debug]

Respond questions from tabular data

options:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Choose witch model will be used from HuggingFace. Default Mistral 7B Instruct
  -t [{task-1,task-2,all}], --task [{task-1,task-2,all}]
                        Choose witch task will be executed. Default both task (all)
  -b [BATCH_SIZE], --batch-size [BATCH_SIZE]
                        Batch size. Default 10
  -n [NUMBER_ROWS], --number-rows [NUMBER_ROWS]
                        Only execute n rows from the Dataset. Default all rows
  -z, --zip-file        Create zip file
  --debug               Create debug file
```

### Examples

```bash
# Task 1 with Codestral model and debug file
python main.py -m mistralai/Codestral-22B-v0.1 --debug --task task-1
```

```bash
# Task 2 with default model, only first 30 rows from dataset with batch size 5
python main.py --task task-2 -b 5 -n 30
```

```bash
# Task 1 and 2 with Qwen Coder 32B model, final results in zip file
python main.py -m Qwen/Qwen2.5-Coder-32B-Instruct -z
```

## Tested Models

| Model | Accuracy Task 1 | Accuracy Task 2 |
| ----- | --------------- |---------------- |
| [Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1) | 0.65625 | 0.71875 |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 0.503125 | 0.453125 |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 0.425 | 0.515625 |
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | **0.7594** | **0.8438** |
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | **0.85** | **0.8438** |
