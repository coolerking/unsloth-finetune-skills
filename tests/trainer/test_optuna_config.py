"""Tests for optuna_config module."""
import pytest
from unittest.mock import MagicMock, patch


# Mock unsloth and related imports before importing the module
@pytest.fixture(autouse=True)
def mock_unsloth():
    """Mock unsloth and related libraries."""
    with patch.dict('sys.modules', {
        'unsloth': MagicMock(),
        'unsloth.FastLanguageModel': MagicMock(),
        'transformers': MagicMock(),
        'trl': MagicMock(),
        'datasets': MagicMock(),
        'optuna': MagicMock(),
    }):
        yield


def test_get_default_search_space(mock_unsloth):
    """Test get_default_search_space returns expected keys."""
    from unsloth_trainer.optuna_config import get_default_search_space
    space = get_default_search_space()
    assert 'learning_rate' in space
    assert 'lora_rank' in space
    assert 'batch_size' in space
    assert 'lora_alpha' in space
    assert 'lora_dropout' in space
    assert 'num_epochs' in space


def test_create_objective(mock_unsloth):
    """Test create_objective returns a callable."""
    from unsloth_trainer.optuna_config import create_objective
    objective = create_objective(
        train_dataset_path="dummy.jsonl",
        eval_dataset_path="dummy.jsonl",
        base_model="gpt-oss:20b",
        output_dir="/tmp/test"
    )
    assert callable(objective)


def test_sample_params(mock_unsloth):
    """Test sample_params samples from search space correctly."""
    from unsloth_trainer.optuna_config import sample_params, get_default_search_space

    search_space = get_default_search_space()
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 0.001
    mock_trial.suggest_int.return_value = 3
    mock_trial.suggest_categorical.side_effect = [16, 32, 2]

    params = sample_params(mock_trial, search_space)

    assert 'learning_rate' in params
    assert 'lora_rank' in params
    assert 'lora_alpha' in params
    assert 'lora_dropout' in params
    assert 'batch_size' in params
    assert 'num_epochs' in params
