# Fine-Tuning Skills Usage Guide

Complete guide for using the fine-tuning skills to create training datasets and fine-tune LLMs.

## Overview

This suite of 4 skills provides an end-to-end workflow for fine-tuning language models:

1. **unsloth_dataset_creator** - Create training datasets from PDF documents
2. **unsloth_trainer** - Fine-tune models with hyperparameter optimization
3. **unsloth_fine_tuning_orchestrator** - Coordinate the complete workflow
4. **unsloth_auto_improver** - Evaluate models and suggest improvements

## Quick Start

### Complete Workflow (Recommended)

Use the orchestrator to run the entire pipeline:

```python
from unsloth_fine_tuning_orchestrator import run_workflow

result = run_workflow(
    pdf_dir="/path/to/pdfs",
    output_dir="/path/to/output",
    base_model="unsloth/Llama-3.2-3B-Instruct",
    config={
        "use_thinking": True,
        "target_samples": 1000
    },
    llm_provider="groq",
    llm_config={
        "model": "llama-3.1-70b-versatile",
        "api_key": "your-groq-api-key"
    },
    optuna_config={
        "n_trials": 20,
        "max_epochs": 5
    }
)

print(f"Workflow status: {result['status']}")
print(f"Run directory: {result['paths']['run_dir']}")
```

## Individual Skills

### 1. Dataset Creator

Create training datasets from PDF documents.

```python
from unsloth_dataset_creator import create_dataset

result = create_dataset(
    pdf_dir="/path/to/pdfs",
    output_dir="/path/to/output",
    config={
        "use_thinking": True,      # Include thinking field in output
        "target_samples": 1000     # Target number of Q&A pairs
    },
    llm_provider="groq",
    llm_config={
        "model": "llama-3.1-70b-versatile",
        "api_key": "your-api-key",
        "temperature": 0.7,
        "timeout": 120
    }
)

if result["status"] == "success":
    print(f"Created {result['metadata']['total_samples']} samples")
    print(f"Train: {result['metadata']['train_count']}")
    print(f"Eval: {result['metadata']['eval_count']}")
```

**Output Files:**
- `dataset_train.jsonl` - Training data (90%)
- `dataset_eval.jsonl` - Evaluation data (10%)
- `dataset_metadata.json` - Generation statistics

**JSONL Format:**
```json
{
  "instruction": "What is the vacation policy?",
  "output": "Employees receive 20 days of paid vacation per year...",
  "thinking": "The user is asking about vacation policy from the employee handbook...",
  "metadata": {
    "source_file": "handbook.pdf",
    "category": "hr",
    "chunk_id": "handbook.pdf_sec5",
    "question_type": "事実確認型",
    "token_count": 245
  }
}
```

### 2. Trainer

Fine-tune models with Optuna hyperparameter search.

```python
from unsloth_trainer import fine_tune

result = fine_tune(
    train_dataset="/path/to/dataset_train.jsonl",
    eval_dataset="/path/to/dataset_eval.jsonl",
    output_dir="/path/to/output",
    base_model="unsloth/Llama-3.2-3B-Instruct",
    optuna_config={
        "n_trials": 20,              # Number of hyperparameter trials
        "max_epochs": 5,             # Max epochs per trial
        "lora_ranks": [8, 16, 32],   # LoRA ranks to try
        "learning_rates": [1e-5, 5e-5, 1e-4]
    }
)

print(f"Best eval loss: {result['best_eval_loss']}")
print(f"Best model: {result['best_model_path']}")
print(f"Best params: {result['best_params']}")
```

**Output Structure:**
```
output_dir/
├── best_model/              # Best performing model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── trials/
│   ├── trial_0/
│   │   └── final_model/
│   ├── trial_1/
│   │   └── final_model/
│   └── ...
└── optuna_study.json        # Optimization results
```

**Hyperparameters Search Space:**

| Parameter | Type | Range |
|-----------|------|-------|
| learning_rate | float | 1e-5 - 1e-3 (log) |
| lora_rank | categorical | 8, 16, 32, 64 |
| lora_alpha | categorical | 16, 32, 64, 128 |
| lora_dropout | float | 0.0 - 0.1 |
| batch_size | categorical | 1, 2, 4 |
| num_epochs | int | 1 - 5 |

### 3. Auto Improver

Evaluate models and generate improvement plans.

```python
from unsloth_auto_improver import evaluate_and_improve

result = evaluate_and_improve(
    model_path="/path/to/best_model",
    eval_dataset="/path/to/dataset_eval.jsonl",
    metric="exact_match",        # or "contains", "fuzzy"
    threshold=0.8,               # Target score
    max_iterations=5             # Max improvement iterations
)

if result["passed"]:
    print(f"Model passed with score: {result['score']}")
else:
    print(f"Model needs improvement. Score: {result['score']}")
    print("Improvement plan:", result["improvement_plan"])
```

**Evaluation Metrics:**

| Metric | Description | Use Case |
|--------|-------------|----------|
| `exact_match` | Case-insensitive exact match | QA with precise answers |
| `contains` | Reference contained in prediction | Long-form answers |
| `fuzzy` | Token overlap > threshold | Paraphrased answers |

**Improvement Plan Output:**
```python
{
    "priority_actions": [
        "Add more procedural/how-to Q&A samples",
        "Increase dataset diversity"
    ],
    "suggested_config_changes": {
        "target_samples": 2000
    },
    "dataset_recommendations": [
        "Focus on 'procedure' question types"
    ],
    "estimated_impact": "High - addressing failure patterns could improve score by 15-20%"
}
```

## Configuration Reference

### Dataset Creator Config

```python
config = {
    "use_thinking": True,      # Include thinking process in output
    "target_samples": 1000     # Target Q&A pairs to generate
}
```

### LLM Config

```python
llm_config = {
    "model": "llama-3.1-70b-versatile",  # Groq model name
    "api_key": "gsk_...",                 # Groq API key
    "temperature": 0.7,                   # Generation temperature
    "timeout": 120                        # Request timeout in seconds
}
```

**Supported Providers:**
- `groq` - Groq Cloud API (default)

### Optuna Config

```python
optuna_config = {
    "n_trials": 20,              # Number of optimization trials
    "max_epochs": 5,             # Max training epochs per trial
    "lora_ranks": [8, 16, 32],   # LoRA ranks to search
    "learning_rates": [1e-5, 5e-5, 1e-4]  # Learning rates to search
}
```

## Error Handling

All skills return structured error responses:

```python
{
    "status": "error",
    "message": "Descriptive error message",
    "error_type": "ValidationError"  # Optional
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No PDF files found` | Empty or invalid pdf_dir | Check directory path |
| `API key required` | Missing Groq API key | Set api_key in llm_config |
| `Dataset file not found` | Missing train/eval files | Check dataset paths |
| `Missing required parameter` | Incomplete optuna_config | Provide all required params |

## Best Practices

### 1. Dataset Creation

- **Organize PDFs by category** in subdirectories for better metadata
- **Start with small target_samples** (100-200) for testing
- **Use use_thinking=True** for complex reasoning tasks

### 2. Training

- **Start with fewer trials** (5-10) to verify the setup
- **Use smaller models** (3B-8B) for faster iteration
- **Monitor disk space** - each trial saves a full model

### 3. Evaluation

- **Choose metric based on task:**
  - Exact facts → `exact_match`
  - Long answers → `contains`
  - Flexible answers → `fuzzy`
- **Set realistic thresholds** based on task difficulty

### 4. Workflow

```python
# 1. Quick test run
result = run_workflow(
    pdf_dir="./pdfs",
    output_dir="./test_run",
    base_model="unsloth/Llama-3.2-3B-Instruct",
    config={"target_samples": 50},
    llm_provider="groq",
    llm_config={"api_key": key, "model": "llama-3.1-8b-instant"},
    optuna_config={"n_trials": 3, "max_epochs": 1}
)

# 2. Full run with best settings
result = run_workflow(
    pdf_dir="./pdfs",
    output_dir="./production_run",
    base_model="unsloth/Llama-3.2-3B-Instruct",
    config={"target_samples": 2000},
    llm_provider="groq",
    llm_config={"api_key": key, "model": "llama-3.1-70b-versatile"},
    optuna_config={"n_trials": 50, "max_epochs": 5}
)
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in optuna_config (try 1)
- Use smaller `lora_rank` values (8 or 16)
- Enable gradient checkpointing (already enabled by default)

### Slow Training

- Use fewer `n_trials` for initial testing
- Reduce `max_epochs` for faster iteration
- Use smaller base models

### Poor Evaluation Scores

- Increase `target_samples` for more training data
- Check failure analysis in auto_improver output
- Adjust `metric` to match task requirements

## Environment Variables

```bash
# Groq API Key (alternative to llm_config)
export GROQ_API_KEY="gsk_..."

# Hugging Face Token (for model downloads)
export HF_TOKEN="hf_..."
```

## Dependencies

```bash
# Core dependencies
pip install unsloth datasets trl transformers
pip install optuna
pip install pdfminer.six tiktoken
pip install requests

# Optional (for better performance)
pip install flash-attn --no-build-isolation
```

## Advanced Usage

### Custom Question Types

Extend `qa_generator.py` to add custom question types:

```python
# In qa_generator.py
QUESTION_TYPES = [
    "事実確認型",
    "手続き型",
    "条件判断型",
    "比較型",
    "具体例型",
    "custom_type"  # Add your type
]

TYPE_TEMPLATES = {
    # ... existing templates
    "custom_type": "Custom instruction for your question type"
}
```

### Custom Metrics

Add custom evaluation metrics:

```python
from unsloth_auto_improver.evaluator import evaluate_model

def my_custom_metric(prediction: str, reference: str) -> bool:
    # Your custom logic
    return similarity_score > 0.9

# Use with evaluate_model
result = evaluate_model(
    model_path="...",
    eval_dataset="...",
    metric="custom"  # Requires modifying evaluator.py
)
```

## API Reference

See individual skill documentation:
- `unsloth_dataset_creator/SKILL.md`
- `unsloth_trainer/SKILL.md`
- `unsloth_fine_tuning_orchestrator/SKILL.md`
- `unsloth_auto_improver/SKILL.md`
