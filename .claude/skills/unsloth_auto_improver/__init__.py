"""Unsloth Auto Improver - Evaluate models and auto-improve through analysis.

This module provides functionality to:
- Evaluate fine-tuned models on evaluation datasets
- Analyze failure patterns from evaluation results
- Generate actionable improvement plans
- Run iterative improvement loops
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .evaluator import evaluate_model, load_eval_dataset

__all__ = [
    'evaluate_and_improve',
    'analyze_failures',
    'generate_improvement_plan',
    'evaluate_model',
    'load_eval_dataset'
]


def analyze_failures(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze failure patterns from evaluation results.

    Examines failed samples to identify common patterns and root causes.

    Args:
        eval_results: Results dictionary from evaluate_model() containing:
            - results: List of per-sample results with 'passed', 'instruction',
                      'reference', 'prediction' keys
            - failed_count: Number of failed samples

    Returns:
        Dictionary containing:
        - total_failures: Total number of failed samples
        - failure_rate: Percentage of samples that failed
        - patterns: Dictionary of identified failure patterns with counts
        - common_errors: List of most common error types
        - length_analysis: Analysis of prediction vs reference lengths
        - sample_failures: List of representative failure examples
    """
    if eval_results.get("status") != "success":
        return {
            "total_failures": 0,
            "failure_rate": 0.0,
            "patterns": {},
            "common_errors": [],
            "length_analysis": {},
            "sample_failures": [],
            "error": eval_results.get("error", "Unknown error")
        }

    results = eval_results.get("results", [])
    failed_count = eval_results.get("failed_count", 0)
    total_count = eval_results.get("total_count", 0)

    failure_rate = failed_count / total_count if total_count > 0 else 0.0

    # Collect failed samples
    failed_samples = [r for r in results if not r.get("passed", True)]

    # Pattern analysis
    patterns = defaultdict(int)
    length_diffs = []
    empty_predictions = 0
    very_long_predictions = 0
    very_short_predictions = 0

    for sample in failed_samples:
        prediction = sample.get("prediction", "")
        reference = sample.get("reference", "")
        instruction = sample.get("instruction", "")

        # Check for empty predictions
        if not prediction or prediction.strip() == "":
            patterns["empty_prediction"] += 1
            empty_predictions += 1
            continue

        # Check for error messages in prediction
        if prediction.startswith("[ERROR:"):
            patterns["inference_error"] += 1
            continue

        # Length analysis
        pred_len = len(prediction.split())
        ref_len = len(reference.split())
        length_diffs.append(pred_len - ref_len)

        if pred_len < ref_len * 0.5:
            patterns["too_short"] += 1
            very_short_predictions += 1
        elif pred_len > ref_len * 2:
            patterns["too_long"] += 1
            very_long_predictions += 1

        # Content patterns
        pred_lower = prediction.lower()
        ref_lower = reference.lower()

        # Check for partial matches
        ref_words = set(ref_lower.split())
        pred_words_set = set(pred_lower.split())
        if ref_words and len(ref_words & pred_words_set) > 0:
            patterns["partial_match"] += 1
        else:
            patterns["no_overlap"] += 1

        # Check for repetitive output
        pred_words_list = pred_lower.split()
        unique_words = len(set(pred_words_list))
        total_words = len(pred_words_list)
        if total_words > 10 and unique_words / total_words < 0.3:
            patterns["repetitive_output"] += 1

        # Check for instruction echoing
        if instruction.lower()[:50] in pred_lower:
            patterns["echoes_instruction"] += 1

        # Check for hallucination markers
        hallucination_markers = ["i think", "maybe", "perhaps", "i believe", "probably"]
        if any(marker in pred_lower for marker in hallucination_markers):
            patterns["uncertain_language"] += 1

    # Calculate length statistics
    length_analysis = {}
    if length_diffs:
        avg_diff = sum(length_diffs) / len(length_diffs)
        length_analysis = {
            "average_length_difference": round(avg_diff, 2),
            "predictions_too_short": very_short_predictions,
            "predictions_too_long": very_long_predictions,
            "empty_predictions": empty_predictions
        }

    # Get top patterns
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    common_errors = [
        {"pattern": p, "count": c, "percentage": round(c / failed_count * 100, 1)}
        for p, c in sorted_patterns[:5]
    ] if failed_count > 0 else []

    # Select representative failure examples (up to 5)
    sample_failures = []
    for sample in failed_samples[:5]:
        sample_failures.append({
            "instruction": sample.get("instruction", "")[:200] + "..." if len(sample.get("instruction", "")) > 200 else sample.get("instruction", ""),
            "reference": sample.get("reference", "")[:200] + "..." if len(sample.get("reference", "")) > 200 else sample.get("reference", ""),
            "prediction": sample.get("prediction", "")[:200] + "..." if len(sample.get("prediction", "")) > 200 else sample.get("prediction", "")
        })

    return {
        "total_failures": failed_count,
        "failure_rate": round(failure_rate * 100, 2),
        "patterns": dict(patterns),
        "common_errors": common_errors,
        "length_analysis": length_analysis,
        "sample_failures": sample_failures
    }


def generate_improvement_plan(
    failure_analysis: Dict[str, Any],
    current_score: float,
    threshold: float
) -> Dict[str, Any]:
    """Generate an actionable improvement plan based on failure analysis.

    Args:
        failure_analysis: Results from analyze_failures()
        current_score: Current evaluation score
        threshold: Target threshold score

    Returns:
        Dictionary containing:
        - current_score: Current evaluation score
        - target_score: Target threshold
        - gap: Difference between target and current score
        - priority_actions: List of recommended actions in priority order
        - suggested_config_changes: Suggested changes to training config
        - dataset_recommendations: Recommendations for dataset improvements
        - estimated_impact: Estimated impact of each recommendation
    """
    gap = threshold - current_score

    priority_actions = []
    suggested_config_changes = {}
    dataset_recommendations = []
    estimated_impact = []

    patterns = failure_analysis.get("patterns", {})
    length_analysis = failure_analysis.get("length_analysis", {})

    # Analyze patterns and generate recommendations
    if patterns.get("empty_prediction", 0) > 0:
        priority_actions.append({
            "priority": "high",
            "action": "Add more diverse training examples with varied response lengths",
            "rationale": "Model is generating empty predictions",
            "pattern_count": patterns["empty_prediction"]
        })
        dataset_recommendations.append("Include examples with varying output lengths")
        estimated_impact.append({
            "action": "Add diverse examples",
            "estimated_score_improvement": "0.05-0.10"
        })

    if patterns.get("inference_error", 0) > 0:
        priority_actions.append({
            "priority": "high",
            "action": "Check model loading and inference configuration",
            "rationale": "Inference errors detected during evaluation",
            "pattern_count": patterns["inference_error"]
        })

    if patterns.get("too_short", 0) > 0 or patterns.get("too_long", 0) > 0:
        priority_actions.append({
            "priority": "medium",
            "action": "Adjust training to better match target output lengths",
            "rationale": f"Length mismatch detected: {length_analysis.get('predictions_too_short', 0)} too short, {length_analysis.get('predictions_too_long', 0)} too long",
            "pattern_count": patterns.get("too_short", 0) + patterns.get("too_long", 0)
        })
        suggested_config_changes["max_seq_length"] = "Consider increasing to allow longer outputs"
        estimated_impact.append({
            "action": "Adjust output length",
            "estimated_score_improvement": "0.03-0.08"
        })

    if patterns.get("no_overlap", 0) > patterns.get("partial_match", 0):
        priority_actions.append({
            "priority": "high",
            "action": "Increase training epochs or add more relevant training data",
            "rationale": "Model outputs have no semantic overlap with references",
            "pattern_count": patterns.get("no_overlap", 0)
        })
        suggested_config_changes["num_train_epochs"] = "Increase by 2-3 epochs"
        dataset_recommendations.append("Add more domain-specific training examples")
        estimated_impact.append({
            "action": "Increase training",
            "estimated_score_improvement": "0.10-0.20"
        })

    if patterns.get("partial_match", 0) > 0:
        priority_actions.append({
            "priority": "medium",
            "action": "Fine-tune with higher quality examples or longer training",
            "rationale": "Model is partially correct but missing key details",
            "pattern_count": patterns["partial_match"]
        })
        dataset_recommendations.append("Add examples with more complete answers")
        estimated_impact.append({
            "action": "Improve example quality",
            "estimated_score_improvement": "0.05-0.15"
        })

    if patterns.get("repetitive_output", 0) > 0:
        priority_actions.append({
            "priority": "medium",
            "action": "Add repetition penalty during inference or diversify training data",
            "rationale": "Model is generating repetitive outputs",
            "pattern_count": patterns["repetitive_output"]
        })
        suggested_config_changes["repetition_penalty"] = "Set to 1.1-1.2"
        estimated_impact.append({
            "action": "Add repetition penalty",
            "estimated_score_improvement": "0.02-0.05"
        })

    if patterns.get("echoes_instruction", 0) > 0:
        priority_actions.append({
            "priority": "low",
            "action": "Add more training examples that don't echo the input",
            "rationale": "Model is echoing parts of the instruction",
            "pattern_count": patterns["echoes_instruction"]
        })
        dataset_recommendations.append("Filter out examples that echo input")

    if patterns.get("uncertain_language", 0) > 0:
        priority_actions.append({
            "priority": "low",
            "action": "Add confident, authoritative training examples",
            "rationale": "Model is using uncertain/hallucinatory language",
            "pattern_count": patterns["uncertain_language"]
        })
        dataset_recommendations.append("Add examples with confident, factual tone")

    # If no specific patterns identified but score is low
    if not priority_actions and gap > 0:
        priority_actions.append({
            "priority": "high",
            "action": "General model improvement needed",
            "rationale": f"Score {current_score:.2f} below threshold {threshold:.2f}",
            "pattern_count": failure_analysis.get("total_failures", 0)
        })
        suggested_config_changes["learning_rate"] = "Try 1e-4 or 5e-5"
        suggested_config_changes["num_train_epochs"] = "Increase to 5-10"
        dataset_recommendations.append("Increase dataset size and diversity")
        estimated_impact.append({
            "action": "General improvements",
            "estimated_score_improvement": "0.10-0.25"
        })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    priority_actions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return {
        "current_score": round(current_score, 4),
        "target_score": threshold,
        "gap": round(gap, 4),
        "priority_actions": priority_actions,
        "suggested_config_changes": suggested_config_changes,
        "dataset_recommendations": dataset_recommendations,
        "estimated_impact": estimated_impact
    }


def evaluate_and_improve(
    model_path: str,
    eval_dataset: str,
    metric: str = "exact_match",
    threshold: float = 0.8,
    max_iterations: int = 5,
    improvement_config: Optional[Dict] = None,
    iteration: int = 1
) -> Dict[str, Any]:
    """Main entry point for evaluate-and-improve workflow.

    Evaluates a model, analyzes failures, and generates an improvement plan.
    If the score is below threshold, provides recommendations for improvement.

    Args:
        model_path: Path to the model directory or HuggingFace model name.
        eval_dataset: Path to the JSONL evaluation dataset.
        metric: Metric to use for evaluation (default: "exact_match").
        threshold: Target score threshold (default: 0.8).
        max_iterations: Maximum number of improvement iterations (default: 5).
        improvement_config: Optional configuration for improvement suggestions.
        iteration: Current iteration number (default: 1).

    Returns:
        Dictionary containing:
        - status: "success" if score >= threshold, "needs_improvement" otherwise
        - iteration: Current iteration number
        - evaluation: Full evaluation results
        - failure_analysis: Analysis of failure patterns
        - improvement_plan: Actionable improvement plan
        - next_steps: Recommended next steps
    """
    improvement_config = improvement_config or {}

    # Run evaluation
    eval_results = evaluate_model(
        model_path=model_path,
        eval_dataset_path=eval_dataset,
        metric=metric,
        max_samples=improvement_config.get("max_samples")
    )

    if eval_results.get("status") != "success":
        return {
            "status": "error",
            "iteration": iteration,
            "error": eval_results.get("error", "Evaluation failed"),
            "evaluation": eval_results
        }

    score = eval_results.get("score", 0.0)

    # Analyze failures
    failure_analysis = analyze_failures(eval_results)

    # Generate improvement plan
    improvement_plan = generate_improvement_plan(
        failure_analysis=failure_analysis,
        current_score=score,
        threshold=threshold
    )

    # Determine status and next steps
    if score >= threshold:
        status = "success"
        next_steps = ["Model meets threshold. Ready for deployment."]
    elif iteration >= max_iterations:
        status = "max_iterations_reached"
        next_steps = [
            f"Maximum iterations ({max_iterations}) reached.",
            "Consider manual review of the model and dataset.",
            "Current improvement plan may need significant changes."
        ]
    else:
        status = "needs_improvement"
        next_steps = [
            f"Iteration {iteration}/{max_iterations}: Score {score:.2f} below threshold {threshold:.2f}",
            "Review the improvement plan and apply suggested changes.",
            "Re-run evaluation after making improvements."
        ]

        # Add specific actions from improvement plan
        for action in improvement_plan.get("priority_actions", [])[:3]:
            next_steps.append(f"[{action['priority'].upper()}] {action['action']}")

    return {
        "status": status,
        "iteration": iteration,
        "score": score,
        "threshold": threshold,
        "evaluation": eval_results,
        "failure_analysis": failure_analysis,
        "improvement_plan": improvement_plan,
        "next_steps": next_steps
    }
