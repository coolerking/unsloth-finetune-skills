"""Integration test for the complete fine-tuning workflow.

This module tests the end-to-end workflow orchestration including:
- Dataset creation via unsloth_dataset_creator
- Model fine-tuning via unsloth_trainer
- Workflow orchestration via unsloth_fine_tuning_orchestrator
"""
import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add skills to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "skills"))

from unsloth_fine_tuning_orchestrator import run_workflow


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Create a temporary PDF directory with a mock PDF file."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    # Create a mock PDF file (just a text file for testing)
    category_dir = pdf_dir / "regulations"
    category_dir.mkdir()
    pdf_file = category_dir / "test_doc.pdf"
    pdf_file.write_text("Mock PDF content for testing")

    return str(pdf_dir)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def sample_config():
    """Return sample configuration for dataset creation."""
    return {
        "use_thinking": True,
        "target_samples": 10
    }


@pytest.fixture
def sample_llm_config():
    """Return sample LLM configuration."""
    return {
        "api_key": "test_api_key",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.7,
        "max_tokens": 2048
    }


@pytest.fixture
def mock_dataset_result(temp_output_dir):
    """Return a mock successful dataset result."""
    dataset_dir = Path(temp_output_dir) / "dataset"
    return {
        "status": "success",
        "message": "Successfully created dataset with 10 samples",
        "metadata": {
            "total_samples": 10,
            "train_count": 9,
            "eval_count": 1,
            "source_documents": 1,
            "total_chunks": 5
        }
    }


@pytest.fixture
def mock_training_result(temp_output_dir):
    """Return a mock successful training result."""
    return {
        "status": "success",
        "best_model_path": str(Path(temp_output_dir) / "training" / "best_model"),
        "best_params": {
            "learning_rate": 5e-5,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 2,
            "num_epochs": 3
        },
        "best_eval_loss": 0.5,
        "output_dir": str(Path(temp_output_dir) / "training"),
        "study_file": str(Path(temp_output_dir) / "training" / "optuna_study.json")
    }


# =============================================================================
# Full Workflow Integration Tests
# =============================================================================

@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_full_workflow_integration(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test the complete workflow from dataset creation to model training.

    This test verifies that:
    1. run_workflow() orchestrates all steps correctly
    2. Dataset creation is called with correct arguments
    3. Training is called with correct arguments
    4. Result contains all expected fields
    5. Workflow completes successfully
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify result structure
    assert result["status"] == "success"
    assert "run_id" in result
    assert "paths" in result
    assert "metadata" in result
    assert "dataset_result" in result
    assert "training_result" in result
    assert "evaluation_result" in result

    # Verify paths
    assert "run_dir" in result["paths"]
    assert "dataset_dir" in result["paths"]
    assert "training_dir" in result["paths"]
    assert "evaluation_dir" in result["paths"]

    # Verify metadata
    assert result["metadata"]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"
    assert result["metadata"]["pdf_dir"] == temp_pdf_dir
    assert result["metadata"]["config"] == sample_config
    assert result["metadata"]["llm_provider"] == "groq"

    # Verify dataset was called with correct arguments
    mock_create_dataset.assert_called_once()
    call_args = mock_create_dataset.call_args
    assert call_args.kwargs["pdf_dir"] == temp_pdf_dir
    assert "output_dir" in call_args.kwargs
    assert call_args.kwargs["config"] == sample_config
    assert call_args.kwargs["llm_provider"] == "groq"
    assert call_args.kwargs["llm_config"] == sample_llm_config

    # Verify training was called with correct arguments
    mock_fine_tune.assert_called_once()
    call_args = mock_fine_tune.call_args
    assert "train_dataset" in call_args.kwargs
    assert "eval_dataset" in call_args.kwargs
    assert "output_dir" in call_args.kwargs
    assert call_args.kwargs["base_model"] == "unsloth/Llama-3.2-3B-Instruct"

    # Verify result contains mock data
    assert result["dataset_result"] == mock_dataset_result
    assert result["training_result"] == mock_training_result
    assert result["evaluation_result"]["status"] == "placeholder"


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_with_dataset_failure(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config
):
    """Test that training is skipped when dataset creation fails.

    This test verifies that:
    1. When dataset creation fails, training is not attempted
    2. Error status is returned with appropriate message
    3. Dataset error details are included in result
    """
    # Setup mock to return failure
    mock_create_dataset.return_value = {
        "status": "error",
        "message": "No PDF files found in directory",
        "metadata": {}
    }

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify error handling
    assert result["status"] == "error"
    assert "Dataset creation failed" in result["error"]
    assert "No PDF files found" in result["error"]

    # Verify training was NOT called
    mock_fine_tune.assert_not_called()

    # Verify dataset result is included
    assert result["dataset_result"] is not None
    assert result["dataset_result"]["status"] == "error"

    # Verify training result is None (not attempted)
    assert result["training_result"] is None


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_with_training_failure(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result
):
    """Test proper error handling when training fails.

    This test verifies that:
    1. When training fails, error is properly propagated
    2. Error status is returned with training error details
    3. Dataset result is preserved in output
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = {
        "status": "error",
        "error": "CUDA out of memory",
        "output_dir": str(Path(temp_output_dir) / "training")
    }

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify error handling
    assert result["status"] == "error"
    assert "Training failed" in result["error"]
    assert "CUDA out of memory" in result["error"]

    # Verify both steps were called
    mock_create_dataset.assert_called_once()
    mock_fine_tune.assert_called_once()

    # Verify dataset result is preserved
    assert result["dataset_result"] == mock_dataset_result

    # Verify training result contains error
    assert result["training_result"] is not None
    assert result["training_result"]["status"] == "error"


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_with_dataset_exception(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config
):
    """Test error handling when dataset creation raises an exception.

    This test verifies that:
    1. Exceptions in dataset creation are caught and handled
    2. Error message includes exception details
    3. Training is not attempted
    """
    # Setup mock to raise exception
    mock_create_dataset.side_effect = Exception("PDF extraction failed: corrupted file")

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify error handling
    assert result["status"] == "error"
    assert "Dataset creation failed" in result["error"]
    assert "PDF extraction failed" in result["error"]

    # Verify training was NOT called
    mock_fine_tune.assert_not_called()


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_with_training_exception(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result
):
    """Test error handling when training raises an exception.

    This test verifies that:
    1. Exceptions in training are caught and handled
    2. Error message includes exception details
    3. Dataset result is preserved
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.side_effect = Exception("Model loading failed: invalid model name")

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify error handling
    assert result["status"] == "error"
    assert "Training failed" in result["error"]
    assert "Model loading failed" in result["error"]

    # Verify dataset result is preserved
    assert result["dataset_result"] == mock_dataset_result


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_with_optuna_config(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test workflow with custom Optuna configuration.

    This test verifies that:
    1. Optuna config is passed to fine_tune correctly
    2. Custom hyperparameter search space is used
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    # Custom Optuna config
    optuna_config = {
        "n_trials": 10,
        "max_epochs": 3,
        "lora_ranks": [8, 16],
        "learning_rates": [1e-5, 5e-5]
    }

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config,
        optuna_config=optuna_config
    )

    # Verify success
    assert result["status"] == "success"

    # Verify fine_tune was called with optuna_config
    mock_fine_tune.assert_called_once()
    call_args = mock_fine_tune.call_args
    assert call_args.kwargs["optuna_config"] == optuna_config


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_creates_run_directory(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that workflow creates proper run directory structure.

    This test verifies that:
    1. Run directory is created with timestamp
    2. Subdirectories are created for dataset, training, evaluation
    3. Run ID is included in result
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify run directory was created
    run_dir = Path(result["paths"]["run_dir"])
    assert run_dir.exists()
    assert run_dir.is_dir()

    # Verify subdirectories exist
    assert (run_dir / "dataset").exists()
    assert (run_dir / "training").exists()
    assert (run_dir / "evaluation").exists()

    # Verify run_id format (YYYYMMDD_HHMMSS)
    run_id = result["run_id"]
    assert len(run_id) == 15  # YYYYMMDD_HHMMSS
    assert run_id[8] == "_"
    assert run_id[:8].isdigit()
    assert run_id[9:].isdigit()


@patch("unsloth_fine_tuning_orchestrator.create_dataset")
@patch("unsloth_fine_tuning_orchestrator.fine_tune")
def test_workflow_dataset_paths_passed_to_training(
    mock_fine_tune,
    mock_create_dataset,
    temp_pdf_dir,
    temp_output_dir,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that dataset output paths are correctly passed to training.

    This test verifies that:
    1. Dataset directory path is constructed correctly
    2. Train and eval dataset paths are passed to fine_tune
    3. Paths use correct filenames (dataset_train.jsonl, dataset_eval.jsonl)
    """
    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    # Execute workflow
    result = run_workflow(
        pdf_dir=temp_pdf_dir,
        output_dir=temp_output_dir,
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify fine_tune was called with correct dataset paths
    mock_fine_tune.assert_called_once()
    call_args = mock_fine_tune.call_args

    # Check train and eval dataset paths
    train_dataset = call_args.kwargs["train_dataset"]
    eval_dataset = call_args.kwargs["eval_dataset"]

    assert "dataset_train.jsonl" in train_dataset
    assert "dataset_eval.jsonl" in eval_dataset
    assert result["paths"]["dataset_dir"] in train_dataset
    assert result["paths"]["dataset_dir"] in eval_dataset
