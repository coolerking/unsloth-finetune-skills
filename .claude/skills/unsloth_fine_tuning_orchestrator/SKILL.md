---
name: unsloth-fine-tuning-orchestrator
description: Orchestrate the complete fine-tuning workflow. Interactive dialogue for configuration, manages dataset creation, training, and auto-improvement loop.
---

# unsloth-fine-tuning-orchestrator

Orchestrates the complete fine-tuning workflow from PDFs to trained model.

## Usage

User provides natural language instruction:
```
「/workspace/s3_data/pdfs/ のPDFからデータセットを作って、
 gpt-oss:20bをファインチューニングして」
```

## Workflow

1. **Workflow Detection**: Determine execution mode
2. **Interactive Configuration**: Model purpose, RAG, Thinking, Tools
3. **Execute Child Skills**: dataset-creator, trainer, auto-improver
4. **Auto-Improvement Loop**: Evaluate and re-train if needed
5. **Final Report**: Success/failure with details

## Outputs

- Complete run directory under `/workspace/outputs/{run_id}/`
- Trained model (if successful)
- Evaluation results
- Complete logs
