---
name: unsloth-dataset-creator
description: Create training datasets from PDFs for fine-tuning. Extracts text, chunks it, generates Q&A pairs using LLM, validates quality, and outputs JSONL format.
---

# unsloth-dataset-creator

Creates fine-tuning datasets from PDF documents.

## Usage

```json
{
  "pdf_dir": "/path/to/pdfs",
  "output_dir": "/path/to/output",
  "config": {
    "use_rag": false,
    "use_thinking": true,
    "use_tools": false,
    "target_samples": 10000
  },
  "llm_provider": "groq",
  "llm_config": {
    "model": "unsloth/Llama-3.2-3B-Instruct",
    "api_key": "${GROQ_API_KEY}",
    "base_url": "..."
  }
}
```

## Outputs

- `dataset_train.jsonl`: Training data (90%)
- `dataset_eval.jsonl`: Evaluation data (10%)
- `dataset_metadata.json`: Generation statistics and quality report

## Process

1. PDF text extraction (pdfminer.six)
2. Section-based chunking
3. LLM-based Q&A generation (5 question types)
4. Quality validation and deduplication
5. Train/eval split
