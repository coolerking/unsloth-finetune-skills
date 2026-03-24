"""Tests for unsloth_auto_improver evaluation functionality."""
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the skills directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "skills"))

from unsloth_auto_improver.evaluator import (
    load_eval_dataset,
    exact_match,
    contains_match,
    fuzzy_match,
    evaluate_model
)
from unsloth_auto_improver import (
    analyze_failures,
    generate_improvement_plan,
    evaluate_and_improve
)


class TestLoadEvalDataset:
    """Tests for load_eval_dataset function."""

    def test_load_valid_jsonl(self, tmp_path):
        """Test loading a valid JSONL file."""
        dataset_path = tmp_path / "eval.jsonl"
        samples = [
            {"instruction": "What is 2+2?", "output": "4"},
            {"instruction": "What is 3+3?", "output": "6"}
        ]
        with open(dataset_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + '\n')

        result = load_eval_dataset(str(dataset_path))
        assert len(result) == 2
        assert result[0]["instruction"] == "What is 2+2?"
        assert result[0]["output"] == "4"

    def test_load_empty_lines(self, tmp_path):
        """Test loading JSONL with empty lines."""
        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            f.write('{"instruction": "Q1", "output": "A1"}\n')
            f.write('\n')
            f.write('{"instruction": "Q2", "output": "A2"}\n')

        result = load_eval_dataset(str(dataset_path))
        assert len(result) == 2

    def test_file_not_found(self):
        """Test error when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_eval_dataset("/nonexistent/path.jsonl")

    def test_invalid_json(self, tmp_path):
        """Test error on invalid JSON."""
        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            f.write('{"invalid json}\n')

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_dataset(str(dataset_path))


class TestExactMatch:
    """Tests for exact_match metric."""

    def test_exact_match_same(self):
        """Test exact match with identical strings."""
        assert exact_match("Hello World", "Hello World") == 1.0

    def test_exact_match_different(self):
        """Test exact match with different strings."""
        assert exact_match("Hello World", "Goodbye World") == 0.0

    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive."""
        assert exact_match("Hello World", "hello world") == 1.0
        assert exact_match("HELLO", "hello") == 1.0

    def test_exact_match_whitespace(self):
        """Test exact match handles whitespace."""
        assert exact_match("  Hello World  ", "Hello World") == 1.0

    def test_exact_match_empty(self):
        """Test exact match with empty strings."""
        assert exact_match("", "") == 1.0
        assert exact_match("", "something") == 0.0


class TestContainsMatch:
    """Tests for contains_match metric."""

    def test_contains_match_true(self):
        """Test when reference is contained in prediction."""
        assert contains_match("The answer is 42", "42") == 1.0
        assert contains_match("Hello World", "World") == 1.0

    def test_contains_match_false(self):
        """Test when reference is not contained."""
        assert contains_match("Hello World", "Goodbye") == 0.0

    def test_contains_match_case_insensitive(self):
        """Test contains match is case-insensitive."""
        assert contains_match("The Answer Is 42", "answer is") == 1.0

    def test_contains_match_whitespace(self):
        """Test contains match handles whitespace."""
        assert contains_match("  The answer is 42  ", "answer is") == 1.0

    def test_contains_match_empty(self):
        """Test contains match with empty strings."""
        assert contains_match("something", "") == 1.0
        assert contains_match("", "something") == 0.0


class TestFuzzyMatch:
    """Tests for fuzzy_match metric."""

    def test_fuzzy_match_identical(self):
        """Test fuzzy match with identical strings."""
        assert fuzzy_match("hello world", "hello world") == 1.0

    def test_fuzzy_match_similar(self):
        """Test fuzzy match with similar strings."""
        # High overlap should pass with default threshold
        assert fuzzy_match("the quick brown fox", "the quick brown fox jumps") == 1.0

    def test_fuzzy_match_different(self):
        """Test fuzzy match with different strings."""
        # No overlap should fail
        assert fuzzy_match("abc", "xyz") == 0.0

    def test_fuzzy_match_threshold(self):
        """Test fuzzy match respects threshold."""
        # Partial overlap - may pass or fail depending on threshold
        result_high = fuzzy_match("hello world", "hello", threshold=0.9)
        result_low = fuzzy_match("hello world", "hello", threshold=0.3)
        assert result_high == 0.0  # Should fail with high threshold
        assert result_low == 1.0   # Should pass with low threshold

    def test_fuzzy_match_empty(self):
        """Test fuzzy match with empty strings."""
        assert fuzzy_match("", "") == 1.0
        assert fuzzy_match("", "something") == 0.0
        assert fuzzy_match("something", "") == 0.0


class TestEvaluateModel:
    """Tests for evaluate_model function with mocked transformers."""

    @patch('unsloth_auto_improver.evaluator._load_model_and_tokenizer')
    @patch('unsloth_auto_improver.evaluator._run_inference')
    def test_evaluate_model_success(self, mock_run_inference, mock_load_model, tmp_path):
        """Test successful model evaluation."""
        # Setup mocks
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_run_inference.return_value = "The answer is 42"

        # Create eval dataset
        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            f.write(json.dumps({"instruction": "What is 2+2?", "output": "4"}) + '\n')
            f.write(json.dumps({"instruction": "What is 3+3?", "output": "6"}) + '\n')

        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            metric="exact_match"
        )

        assert result["status"] == "success"
        assert "score" in result
        assert "passed_count" in result
        assert "failed_count" in result
        assert "results" in result
        assert len(result["results"]) == 2

    @patch('unsloth_auto_improver.evaluator._load_model_and_tokenizer')
    @patch('unsloth_auto_improver.evaluator._run_inference')
    def test_evaluate_model_with_metrics(self, mock_run_inference, mock_load_model, tmp_path):
        """Test evaluation with different metrics."""
        # Setup mocks
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_run_inference.return_value = "42"

        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            f.write(json.dumps({"instruction": "Q1", "output": "42"}) + '\n')

        # Test exact_match
        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            metric="exact_match"
        )
        assert result["metric"] == "exact_match"

        # Test contains_match
        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            metric="contains_match"
        )
        assert result["metric"] == "contains_match"

        # Test fuzzy_match
        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            metric="fuzzy_match"
        )
        assert result["metric"] == "fuzzy_match"

    def test_evaluate_model_invalid_metric(self, tmp_path):
        """Test error with invalid metric."""
        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            f.write(json.dumps({"instruction": "Q1", "output": "A1"}) + '\n')

        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            metric="invalid_metric"
        )

        assert result["status"] == "error"
        assert "Unsupported metric" in result["error"]

    def test_evaluate_model_missing_dataset(self):
        """Test error when dataset does not exist."""
        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path="/nonexistent/eval.jsonl"
        )

        assert result["status"] == "error"

    def test_evaluate_model_empty_dataset(self, tmp_path):
        """Test error with empty dataset."""
        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            pass  # Empty file

        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path)
        )

        assert result["status"] == "error"
        assert "empty" in result["error"].lower()

    @patch('unsloth_auto_improver.evaluator._load_model_and_tokenizer')
    @patch('unsloth_auto_improver.evaluator._run_inference')
    def test_evaluate_model_max_samples(self, mock_run_inference, mock_load_model, tmp_path):
        """Test max_samples parameter limits evaluation."""
        # Setup mocks
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_run_inference.return_value = "answer"

        dataset_path = tmp_path / "eval.jsonl"
        with open(dataset_path, 'w') as f:
            for i in range(10):
                f.write(json.dumps({"instruction": f"Q{i}", "output": f"A{i}"}) + '\n')

        result = evaluate_model(
            model_path="/fake/model",
            eval_dataset_path=str(dataset_path),
            max_samples=3
        )

        assert result["status"] == "success"
        assert result["total_count"] == 3


class TestAnalyzeFailures:
    """Tests for analyze_failures function."""

    def test_analyze_all_passed(self):
        """Test analysis when all samples pass."""
        eval_results = {
            "status": "success",
            "results": [
                {"instruction": "Q1", "reference": "A1", "prediction": "A1", "passed": True},
                {"instruction": "Q2", "reference": "A2", "prediction": "A2", "passed": True}
            ],
            "failed_count": 0,
            "total_count": 2
        }

        analysis = analyze_failures(eval_results)

        assert analysis["total_failures"] == 0
        assert analysis["failure_rate"] == 0.0
        assert analysis["patterns"] == {}

    def test_analyze_with_failures(self):
        """Test analysis with failed samples."""
        eval_results = {
            "status": "success",
            "results": [
                {"instruction": "Q1", "reference": "A1", "prediction": "A1", "passed": True},
                {"instruction": "Q2", "reference": "A2", "prediction": "wrong", "passed": False},
                {"instruction": "Q3", "reference": "A3", "prediction": "", "passed": False}
            ],
            "failed_count": 2,
            "total_count": 3
        }

        analysis = analyze_failures(eval_results)

        assert analysis["total_failures"] == 2
        assert analysis["failure_rate"] == pytest.approx(66.67, 0.1)
        assert "empty_prediction" in analysis["patterns"]

    def test_analyze_error_status(self):
        """Test analysis when evaluation had error."""
        eval_results = {
            "status": "error",
            "error": "Something went wrong"
        }

        analysis = analyze_failures(eval_results)

        assert analysis["total_failures"] == 0
        assert "error" in analysis

    def test_analyze_patterns(self):
        """Test detection of various failure patterns."""
        eval_results = {
            "status": "success",
            "results": [
                {
                    "instruction": "What is the answer?",
                    "reference": "The answer is 42",
                    "prediction": "The answer is 42 is the answer to everything",
                    "passed": False
                },
                {
                    "instruction": "List colors",
                    "reference": "Red, Blue, Green",
                    "prediction": "Red, Blue, Green, Red, Blue, Green, Red, Blue, Green",
                    "passed": False
                }
            ],
            "failed_count": 2,
            "total_count": 2
        }

        analysis = analyze_failures(eval_results)

        assert analysis["total_failures"] == 2
        assert "length_analysis" in analysis
        assert "common_errors" in analysis
        assert len(analysis["sample_failures"]) > 0


class TestGenerateImprovementPlan:
    """Tests for generate_improvement_plan function."""

    def test_plan_meets_threshold(self):
        """Test plan when score meets threshold."""
        failure_analysis = {
            "total_failures": 0,
            "patterns": {}
        }

        plan = generate_improvement_plan(failure_analysis, current_score=0.9, threshold=0.8)

        assert plan["current_score"] == 0.9
        assert plan["target_score"] == 0.8
        # Gap can be negative when above threshold (gap = threshold - current)
        assert plan["current_score"] >= plan["target_score"]

    def test_plan_below_threshold(self):
        """Test plan generation when below threshold."""
        failure_analysis = {
            "total_failures": 10,
            "patterns": {
                "empty_prediction": 3,
                "too_short": 4,
                "no_overlap": 3
            }
        }

        plan = generate_improvement_plan(failure_analysis, current_score=0.5, threshold=0.8)

        assert plan["current_score"] == 0.5
        assert plan["target_score"] == 0.8
        assert plan["gap"] == 0.3
        assert len(plan["priority_actions"]) > 0
        assert len(plan["dataset_recommendations"]) > 0

    def test_plan_with_specific_patterns(self):
        """Test plan addresses specific failure patterns."""
        failure_analysis = {
            "total_failures": 5,
            "patterns": {
                "repetitive_output": 2,
                "echoes_instruction": 1,
                "uncertain_language": 2
            }
        }

        plan = generate_improvement_plan(failure_analysis, current_score=0.6, threshold=0.8)

        actions = [a["action"] for a in plan["priority_actions"]]
        assert any("repetition" in a.lower() for a in actions)


class TestEvaluateAndImprove:
    """Tests for evaluate_and_improve main entry point."""

    @patch('unsloth_auto_improver.evaluate_model')
    def test_success_meets_threshold(self, mock_evaluate):
        """Test when evaluation meets threshold."""
        mock_evaluate.return_value = {
            "status": "success",
            "score": 0.85,
            "passed_count": 85,
            "failed_count": 15,
            "total_count": 100,
            "results": []
        }

        result = evaluate_and_improve(
            model_path="/fake/model",
            eval_dataset="/fake/eval.jsonl",
            threshold=0.8
        )

        assert result["status"] == "success"
        assert result["score"] == 0.85
        assert "ready for deployment" in result["next_steps"][0].lower()

    @patch('unsloth_auto_improver.evaluate_model')
    def test_needs_improvement(self, mock_evaluate):
        """Test when evaluation needs improvement."""
        mock_evaluate.return_value = {
            "status": "success",
            "score": 0.6,
            "passed_count": 60,
            "failed_count": 40,
            "total_count": 100,
            "results": [
                {"instruction": "Q1", "reference": "A1", "prediction": "wrong", "passed": False}
            ]
        }

        result = evaluate_and_improve(
            model_path="/fake/model",
            eval_dataset="/fake/eval.jsonl",
            threshold=0.8,
            max_iterations=5,
            iteration=1
        )

        assert result["status"] == "needs_improvement"
        assert result["iteration"] == 1
        assert "improvement_plan" in result
        assert len(result["next_steps"]) > 1

    @patch('unsloth_auto_improver.evaluate_model')
    def test_max_iterations_reached(self, mock_evaluate):
        """Test when max iterations is reached."""
        mock_evaluate.return_value = {
            "status": "success",
            "score": 0.6,
            "passed_count": 60,
            "failed_count": 40,
            "total_count": 100,
            "results": []
        }

        result = evaluate_and_improve(
            model_path="/fake/model",
            eval_dataset="/fake/eval.jsonl",
            threshold=0.8,
            max_iterations=3,
            iteration=3
        )

        assert result["status"] == "max_iterations_reached"
        assert "manual review" in result["next_steps"][1].lower()

    @patch('unsloth_auto_improver.evaluate_model')
    def test_evaluation_error(self, mock_evaluate):
        """Test handling of evaluation error."""
        mock_evaluate.return_value = {
            "status": "error",
            "error": "Model not found"
        }

        result = evaluate_and_improve(
            model_path="/fake/model",
            eval_dataset="/fake/eval.jsonl"
        )

        assert result["status"] == "error"
        assert "model not found" in result["error"].lower()

    @patch('unsloth_auto_improver.evaluate_model')
    def test_iteration_tracking(self, mock_evaluate):
        """Test iteration number is tracked correctly."""
        mock_evaluate.return_value = {
            "status": "success",
            "score": 0.9,
            "passed_count": 90,
            "failed_count": 10,
            "total_count": 100,
            "results": []
        }

        result = evaluate_and_improve(
            model_path="/fake/model",
            eval_dataset="/fake/eval.jsonl",
            iteration=5
        )

        assert result["iteration"] == 5
