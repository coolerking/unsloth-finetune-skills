---
name: unsloth-auto-improver
description: Evaluate trained models and auto-improve through analysis. Detects dataset issues and suggests improvements until target score is reached.
---

# unsloth-auto-improver

Evaluates fine-tuned models and provides automatic improvement loop.

## Usage

```json
{
  "model_path": "/path/to/model",
  "eval_dataset": "/path/to/eval.jsonl",
  "metric": "rouge_l",
  "threshold": 0.8,
  "max_iterations": 5
}
```

## Outputs

- `eval_results.json`: Per-sample scores
- `analysis_report.md`: Failure analysis and improvement suggestions
- `improvement_plan.json`: Actionable improvement plan

## Process

1. Run model on eval dataset
2. Calculate evaluation metric
3. Compare to threshold
4. If below threshold:
   - Analyze failure patterns
   - Identify root cause
   - Generate improvement plan
5. If dataset issues detected, exit with report
