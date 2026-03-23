---
name: unsloth-trainer
description: Fine-tune models using unsloth with Optuna hyperparameter search. Includes TensorBoard logging and checkpoint management.
---

# unsloth-trainer

Fine-tunes models using unsloth for efficient training with Optuna hyperparameter optimization.

## Usage

```json
{
  "train_dataset": "/path/to/train.jsonl",
  "eval_dataset": "/path/to/eval.jsonl",
  "output_dir": "/path/to/output",
  "base_model": "unsloth/Llama-3.2-3B-Instruct",
  "optuna_config": {
    "n_trials": 20,
    "max_epochs": 5,
    "lora_ranks": [8, 16, 32],
    "learning_rates": [1e-5, 5e-5, 1e-4]
  }
}
```

## Outputs

- `best_model/`: Best performing model (HF format)
- `checkpoints/trial_{n}/`: Per-trial checkpoints
- `optuna_study.json`: Optimization history
- `tensorboard_logs/`: Training logs

## Process

1. Load datasets (JSONL format)
2. Optuna hyperparameter search
3. Train with unsloth (FastLanguageModel)
4. Save best model and checkpoints
