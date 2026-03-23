"""Tests for the unsloth_fine_tuning_orchestrator workflow."""
import pytest
import json
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock


# Mock unsloth and related imports before importing the orchestrator
@pytest.fixture(autouse=True)
def mock_unsloth_and_optuna():
    """Mock unsloth, optuna, and related libraries."""
    with patch.dict('sys.modules', {
        'unsloth': MagicMock(),
        'unsloth.FastLanguageModel': MagicMock(),
        'transformers': MagicMock(),
        'trl': MagicMock(),
        'datasets': MagicMock(),
        'optuna': MagicMock(),
    }):
        # Setup mock optuna
        mock_optuna = sys.modules['optuna']

        # Mock study
        mock_study = MagicMock()
        mock_study.best_params = {
            'learning_rate': 5e-5,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'batch_size': 2,
            'num_epochs': 3
        }
        mock_study.best_value = 0.25
        mock_study.best_trial.number = 2

        # Create trial mocks
        def create_trial_mock(number, params, value):
            trial = MagicMock()
            trial.number = number
            trial.params = params
            trial.value = value
            trial.state = MagicMock()
            trial.state.name = 'COMPLETE'
            return trial

        mock_study.trials = [
            create_trial_mock(0, {'lr': 1e-5}, 0.5),
            create_trial_mock(1, {'lr': 5e-5}, 0.3),
            create_trial_mock(2, {'lr': 1e-4}, 0.25),
        ]

        mock_optuna.create_study.return_value = mock_study
        mock_optuna.samplers.TPESampler = MagicMock

        yield mock_optuna, mock_study


@pytest.fixture
def sample_config():
    """Return sample configuration for testing."""
    return {
        "use_thinking": True,
        "target_samples": 10
    }


@pytest.fixture
def sample_llm_config():
    """Return sample LLM configuration for testing."""
    return {
        "api_key": "test_api_key",
        "model": "llama-3.1-8b-instant",
        "temperature": 0.7,
        "max_tokens": 2048
    }


@pytest.fixture
def sample_optuna_config():
    """Return sample Optuna configuration for testing."""
    return {
        "n_trials": 5,
        "max_epochs": 2,
        "lora_ranks": [8, 16],
        "learning_rates": [1e-5, 5e-5]
    }


@pytest.fixture
def mock_dataset_result():
    """Return mock successful dataset creation result."""
    return {
        "status": "success",
        "message": "Successfully created dataset with 10 samples",
        "metadata": {
            "total_samples": 10,
            "train_count": 9,
            "eval_count": 1,
            "source_documents": 2,
            "total_chunks": 5
        }
    }


@pytest.fixture
def mock_training_result():
    """Return mock successful training result."""
    return {
        "status": "success",
        "best_model_path": "/path/to/best_model",
        "best_params": {
            "learning_rate": 5e-5,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "batch_size": 2,
            "num_epochs": 3
        },
        "best_eval_loss": 0.1234,
        "output_dir": "/path/to/output",
        "study_file": "/path/to/study.json"
    }


# =============================================================================
# create_run_directory Tests
# =============================================================================

def test_create_run_directory(tmp_path, mock_unsloth_and_optuna):
    """Test that run directory is created with correct timestamp format."""
    from unsloth_fine_tuning_orchestrator import create_run_directory

    run_dir = create_run_directory(str(tmp_path))
    run_path = Path(run_dir)

    # Verify directory exists
    assert run_path.exists()
    assert run_path.is_dir()

    # Verify naming format: run_YYYYMMDD_HHMMSS
    assert run_path.name.startswith("run_")
    timestamp_part = run_path.name.replace("run_", "")
    assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
    assert timestamp_part[8] == "_"


def test_create_run_directory_nested(tmp_path, mock_unsloth_and_optuna):
    """Test that run directory is created in nested path."""
    from unsloth_fine_tuning_orchestrator import create_run_directory

    nested_dir = tmp_path / "nested" / "output"
    run_dir = create_run_directory(str(nested_dir))
    run_path = Path(run_dir)

    assert run_path.exists()
    assert run_path.is_dir()
    assert "nested" in str(run_path)


# =============================================================================
# run_workflow Success Tests
# =============================================================================

@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_success(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    sample_optuna_config,
    mock_dataset_result,
    mock_training_result
):
    """Test successful workflow execution."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    # Setup mocks
    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config,
        optuna_config=sample_optuna_config
    )

    # Verify result structure
    assert result['status'] == 'success'
    assert 'run_id' in result
    assert 'paths' in result
    assert 'metadata' in result
    assert 'dataset_result' in result
    assert 'training_result' in result
    assert 'evaluation_result' in result

    # Verify dataset was called correctly
    mock_create_dataset.assert_called_once()
    call_args = mock_create_dataset.call_args
    assert call_args.kwargs['pdf_dir'] == str(pdf_dir)
    assert call_args.kwargs['config'] == sample_config
    assert call_args.kwargs['llm_provider'] == "groq"
    assert call_args.kwargs['llm_config'] == sample_llm_config

    # Verify training was called correctly
    mock_fine_tune.assert_called_once()
    call_args = mock_fine_tune.call_args
    assert 'train_dataset' in call_args.kwargs
    assert 'eval_dataset' in call_args.kwargs
    assert call_args.kwargs['base_model'] == "unsloth/Llama-3.2-3B-Instruct"
    assert call_args.kwargs['optuna_config'] == sample_optuna_config


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_creates_directories(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that workflow creates all necessary directories."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify directories were created
    run_dir = Path(result['paths']['run_dir'])
    assert run_dir.exists()
    assert (run_dir / 'dataset').exists()
    assert (run_dir / 'training').exists()
    assert (run_dir / 'evaluation').exists()


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_without_optuna_config(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test workflow without optional optuna_config."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
        # optuna_config is None (default)
    )

    assert result['status'] == 'success'
    mock_fine_tune.assert_called_once()
    # Verify optuna_config was passed as None
    call_args = mock_fine_tune.call_args
    assert call_args.kwargs['optuna_config'] is None


# =============================================================================
# run_workflow Error Handling Tests
# =============================================================================

@patch('unsloth_fine_tuning_orchestrator.create_dataset')
def test_run_workflow_dataset_failure(
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config
):
    """Test workflow handles dataset creation failure."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = {
        "status": "error",
        "message": "No PDF files found",
        "metadata": {}
    }

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    assert result['status'] == 'error'
    assert 'Dataset creation failed' in result['error']
    assert 'No PDF files found' in result['error']


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
def test_run_workflow_dataset_exception(
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config
):
    """Test workflow handles dataset creation exception."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.side_effect = Exception("PDF extraction failed")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    assert result['status'] == 'error'
    assert 'Dataset creation failed' in result['error']
    assert 'PDF extraction failed' in result['error']


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_training_failure(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result
):
    """Test workflow handles training failure."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = {
        "status": "error",
        "error": "CUDA out of memory"
    }

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    assert result['status'] == 'error'
    assert 'Training failed' in result['error']
    assert 'CUDA out of memory' in result['error']


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_training_exception(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result
):
    """Test workflow handles training exception."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.side_effect = Exception("Model loading failed")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    assert result['status'] == 'error'
    assert 'Training failed' in result['error']
    assert 'Model loading failed' in result['error']


# =============================================================================
# run_workflow Sequence Tests
# =============================================================================

@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_skips_training_on_dataset_failure(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config
):
    """Test that training is skipped if dataset creation fails."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = {
        "status": "error",
        "message": "Failed to extract text",
        "metadata": {}
    }

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify training was NOT called
    mock_fine_tune.assert_not_called()
    assert result['status'] == 'error'


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_correct_sequence(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that workflow steps are executed in correct order."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify dataset was called before training
    assert mock_create_dataset.called
    assert mock_fine_tune.called
    assert mock_create_dataset.call_count == 1
    assert mock_fine_tune.call_count == 1


# =============================================================================
# run_workflow Result Structure Tests
# =============================================================================

@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_result_structure(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that result has correct structure with all expected fields."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify result structure
    assert isinstance(result['run_id'], str)
    assert len(result['run_id']) > 0

    # Verify paths structure
    paths = result['paths']
    assert 'run_dir' in paths
    assert 'dataset_dir' in paths
    assert 'training_dir' in paths
    assert 'evaluation_dir' in paths

    # Verify metadata structure
    metadata = result['metadata']
    assert metadata['base_model'] == "unsloth/Llama-3.2-3B-Instruct"
    assert metadata['pdf_dir'] == str(pdf_dir)
    assert metadata['config'] == sample_config
    assert metadata['llm_provider'] == "groq"

    # Verify evaluation placeholder
    assert result['evaluation_result']['status'] == 'placeholder'


@patch('unsloth_fine_tuning_orchestrator.create_dataset')
@patch('unsloth_fine_tuning_orchestrator.fine_tune')
def test_run_workflow_includes_dataset_paths_in_training(
    mock_fine_tune,
    mock_create_dataset,
    tmp_path,
    mock_unsloth_and_optuna,
    sample_config,
    sample_llm_config,
    mock_dataset_result,
    mock_training_result
):
    """Test that training receives correct dataset paths."""
    from unsloth_fine_tuning_orchestrator import run_workflow

    mock_create_dataset.return_value = mock_dataset_result
    mock_fine_tune.return_value = mock_training_result

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run_workflow(
        pdf_dir=str(pdf_dir),
        output_dir=str(output_dir),
        base_model="unsloth/Llama-3.2-3B-Instruct",
        config=sample_config,
        llm_provider="groq",
        llm_config=sample_llm_config
    )

    # Verify training was called with correct dataset paths
    call_args = mock_fine_tune.call_args
    train_dataset_path = call_args.kwargs['train_dataset']
    eval_dataset_path = call_args.kwargs['eval_dataset']

    assert 'dataset_train.jsonl' in train_dataset_path
    assert 'dataset_eval.jsonl' in eval_dataset_path
    assert result['paths']['dataset_dir'] in train_dataset_path
    assert result['paths']['dataset_dir'] in eval_dataset_path
