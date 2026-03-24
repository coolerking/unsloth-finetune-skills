# Fine-Tuning Skills

A suite of 4 Claude Code skills for end-to-end fine-tuning of language models.

## Skills Overview

| Skill | Purpose | Status |
|-------|---------|--------|
| `unsloth_dataset_creator` | Create training datasets from PDF documents | ✅ Complete |
| `unsloth_trainer` | Fine-tune models with hyperparameter optimization | ✅ Complete |
| `unsloth_fine_tuning_orchestrator` | Coordinate complete fine-tuning workflow | ✅ Complete |
| `unsloth_auto_improver` | Evaluate models and suggest improvements | ✅ Complete |

## Quick Start

```python
from unsloth_fine_tuning_orchestrator import run_workflow

result = run_workflow(
    pdf_dir="./pdfs",
    output_dir="./output",
    base_model="unsloth/Llama-3.2-3B-Instruct",
    config={"target_samples": 1000},
    llm_provider="groq",
    llm_config={"api_key": "your-key", "model": "llama-3.1-70b-versatile"}
)
```

## Installation

```bash
# Install dependencies
pip install unsloth datasets trl transformers
pip install optuna
pip install pdfminer.six tiktoken

# Optional: Install skills package
pip install -e .
```

## Usage

See [docs/fine-tuning-skills-usage.md](../../docs/fine-tuning-skills-usage.md) for detailed usage guide.

## Individual Skills

### Dataset Creator

Extract text from PDFs and generate Q&A pairs:

```python
from unsloth_dataset_creator import create_dataset

result = create_dataset(
    pdf_dir="./pdfs",
    output_dir="./dataset",
    config={"target_samples": 1000},
    llm_provider="groq",
    llm_config={"api_key": "your-key"}
)
```

### Trainer

Fine-tune with hyperparameter search:

```python
from unsloth_trainer import fine_tune

result = fine_tune(
    train_dataset="./dataset_train.jsonl",
    eval_dataset="./dataset_eval.jsonl",
    output_dir="./models",
    base_model="unsloth/Llama-3.2-3B-Instruct"
)
```

### Auto Improver

Evaluate and improve models:

```python
from unsloth_auto_improver import evaluate_and_improve

result = evaluate_and_improve(
    model_path="./models/best_model",
    eval_dataset="./dataset_eval.jsonl",
    metric="exact_match",
    threshold=0.8
)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific skill tests
pytest tests/dataset_creator/ -v
pytest tests/trainer/ -v
pytest tests/orchestrator/ -v
pytest tests/auto_improver/ -v
```

## Directory Structure

```
.claude/skills/
├── unsloth_dataset_creator/
│   ├── __init__.py          # Main entry point
│   ├── pdf_processor.py     # PDF text extraction
│   ├── chunker.py           # Document chunking
│   ├── qa_generator.py      # Q&A generation
│   └── SKILL.md             # Skill documentation
├── unsloth_trainer/
│   ├── __init__.py          # Main entry point
│   ├── training_loop.py     # Training implementation
│   ├── optuna_config.py     # Hyperparameter search
│   └── SKILL.md
├── unsloth_fine_tuning_orchestrator/
│   ├── __init__.py          # Workflow coordination
│   └── SKILL.md
└── unsloth_auto_improver/
    ├── __init__.py          # Main entry point
    ├── evaluator.py         # Evaluation metrics
    └── SKILL.md
```

## Workflow

The orchestrator coordinates the complete workflow:

1. **Dataset Creation**
   - Extract text from PDFs
   - Chunk documents
   - Generate Q&A pairs using LLM
   - Split into train/eval sets

2. **Training**
   - Run Optuna hyperparameter search
   - Train multiple trials
   - Select best model

3. **Evaluation** (Optional)
   - Evaluate best model
   - Analyze failures
   - Generate improvement plan

## Configuration

### Required Environment Variables

```bash
export GROQ_API_KEY="gsk_..."  # For Q&A generation
export HF_TOKEN="hf_..."       # For model downloads (optional)
```

### Dataset Creator Config

```python
config = {
    "use_thinking": True,      # Include thinking field
    "target_samples": 1000     # Target Q&A pairs
}
```

### Optuna Config

```python
optuna_config = {
    "n_trials": 20,              # Number of trials
    "max_epochs": 5,             # Max epochs per trial
    "lora_ranks": [8, 16, 32],   # LoRA ranks to try
    "learning_rates": [1e-5, 5e-5, 1e-4]
}
```

## Outputs

### Dataset Creator

- `dataset_train.jsonl` - Training data (90%)
- `dataset_eval.jsonl` - Evaluation data (10%)
- `dataset_metadata.json` - Generation statistics

### Trainer

- `best_model/` - Best performing model
- `trials/trial_{n}/` - Per-trial checkpoints
- `optuna_study.json` - Optimization results

### Auto Improver

- Evaluation scores
- Failure analysis report
- Improvement recommendations

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size or lora_rank |
| Slow training | Use fewer n_trials or max_epochs |
| API errors | Check GROQ_API_KEY and model name |
| Import errors | Install all dependencies |

See [docs/fine-tuning-skills-usage.md](../../docs/fine-tuning-skills-usage.md) for detailed troubleshooting.

## Development

```bash
# Setup development environment
git clone <repo>
cd <repo>
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check .
black .
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the [usage guide](../../docs/fine-tuning-skills-usage.md)
- Review individual SKILL.md files
- Open an issue on GitHub
