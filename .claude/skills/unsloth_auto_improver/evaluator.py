"""Model evaluation utilities for unsloth_auto_improver.

This module provides functionality to:
- Load evaluation datasets in JSONL format
- Calculate various evaluation metrics (exact_match, contains_match, fuzzy_match)
- Run model inference and evaluate performance
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_eval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file.

    Args:
        dataset_path: Path to the JSONL evaluation dataset.

    Returns:
        List of dictionaries containing evaluation samples.
        Each sample should have 'instruction' (input) and 'output' (reference) keys.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset contains invalid JSON.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

    return samples


def exact_match(prediction: str, reference: str) -> float:
    """Calculate case-insensitive exact match score.

    Args:
        prediction: The predicted output from the model.
        reference: The reference (ground truth) output.

    Returns:
        1.0 if prediction matches reference (case-insensitive), 0.0 otherwise.
    """
    pred_normalized = prediction.strip().lower()
    ref_normalized = reference.strip().lower()
    return 1.0 if pred_normalized == ref_normalized else 0.0


def contains_match(prediction: str, reference: str) -> float:
    """Check if reference is contained within prediction.

    Args:
        prediction: The predicted output from the model.
        reference: The reference (ground truth) output.

    Returns:
        1.0 if reference is contained in prediction (case-insensitive), 0.0 otherwise.
    """
    pred_normalized = prediction.strip().lower()
    ref_normalized = reference.strip().lower()
    return 1.0 if ref_normalized in pred_normalized else 0.0


def fuzzy_match(prediction: str, reference: str, threshold: float = 0.8) -> float:
    """Calculate token overlap based fuzzy match score.

    Uses token set overlap to compute similarity between prediction and reference.

    Args:
        prediction: The predicted output from the model.
        reference: The reference (ground truth) output.
        threshold: Minimum overlap ratio to consider a match (default: 0.8).

    Returns:
        1.0 if token overlap ratio >= threshold, 0.0 otherwise.
    """
    pred_tokens = set(prediction.strip().lower().split())
    ref_tokens = set(reference.strip().lower().split())

    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0

    if not pred_tokens:
        return 0.0

    intersection = pred_tokens & ref_tokens
    union = pred_tokens | ref_tokens

    overlap_ratio = len(intersection) / len(union)
    return 1.0 if overlap_ratio >= threshold else 0.0


def _load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from path.

    Args:
        model_path: Path to the model directory or HuggingFace model name.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        Exception: If model loading fails.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer


def _run_inference(model, tokenizer, instruction: str, device: str = "auto") -> str:
    """Run model inference on a single instruction.

    Args:
        model: The loaded transformers model.
        tokenizer: The loaded transformers tokenizer.
        instruction: The input instruction/prompt.
        device: Device to run inference on (default: "auto").

    Returns:
        The model's generated output as a string.
    """
    from transformers import pipeline

    # Create a text generation pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Generate response
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = generator(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False
    )

    return outputs[0]["generated_text"].strip()


def evaluate_model(
    model_path: str,
    eval_dataset_path: str,
    metric: str = "exact_match",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model on an evaluation dataset.

    Loads the model, runs inference on evaluation samples, and calculates
    the specified metric.

    Args:
        model_path: Path to the model directory or HuggingFace model name.
        eval_dataset_path: Path to the JSONL evaluation dataset.
        metric: Metric to use for evaluation. One of:
            - "exact_match": Case-insensitive exact match
            - "contains_match": Reference contained in prediction
            - "fuzzy_match": Token overlap with 0.8 threshold
        max_samples: Maximum number of samples to evaluate (None for all).

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - score: Overall score (average of all sample scores)
        - passed_count: Number of samples that passed (score == 1.0)
        - failed_count: Number of samples that failed (score < 1.0)
        - total_count: Total number of samples evaluated
        - metric: The metric used for evaluation
        - results: List of per-sample results with keys:
            - instruction: Input prompt
            - reference: Expected output
            - prediction: Model's output
            - score: Score for this sample (0.0 or 1.0)
            - passed: Boolean indicating if sample passed
        - error: Error message (if status is "error")

    Raises:
        ValueError: If an unsupported metric is specified.
        FileNotFoundError: If model or dataset paths do not exist.
    """
    # Validate metric
    metric_funcs = {
        "exact_match": exact_match,
        "contains_match": contains_match,
        "fuzzy_match": fuzzy_match
    }

    if metric not in metric_funcs:
        return {
            "status": "error",
            "error": f"Unsupported metric: {metric}. Supported: {list(metric_funcs.keys())}"
        }

    metric_func = metric_funcs[metric]

    try:
        # Load evaluation dataset
        samples = load_eval_dataset(eval_dataset_path)

        if not samples:
            return {
                "status": "error",
                "error": "Evaluation dataset is empty"
            }

        # Limit samples if specified
        if max_samples is not None:
            samples = samples[:max_samples]

        # Load model and tokenizer
        try:
            model, tokenizer = _load_model_and_tokenizer(model_path)
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load model: {str(e)}"
            }

        # Run evaluation
        results = []
        passed_count = 0
        failed_count = 0

        for sample in samples:
            instruction = sample.get("instruction", "")
            reference = sample.get("output", "")

            if not instruction:
                continue

            try:
                prediction = _run_inference(model, tokenizer, instruction)
                score = metric_func(prediction, reference)
                passed = score >= 1.0

                if passed:
                    passed_count += 1
                else:
                    failed_count += 1

                results.append({
                    "instruction": instruction,
                    "reference": reference,
                    "prediction": prediction,
                    "score": score,
                    "passed": passed
                })
            except Exception as e:
                # Record failed sample
                failed_count += 1
                results.append({
                    "instruction": instruction,
                    "reference": reference,
                    "prediction": f"[ERROR: {str(e)}]",
                    "score": 0.0,
                    "passed": False
                })

        total_count = passed_count + failed_count
        overall_score = passed_count / total_count if total_count > 0 else 0.0

        return {
            "status": "success",
            "score": overall_score,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": total_count,
            "metric": metric,
            "results": results
        }

    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Evaluation failed: {str(e)}"
        }
