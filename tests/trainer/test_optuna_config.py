from unittest.mock import MagicMock
from skills.unsloth_trainer.optuna_config import get_default_search_space, create_objective


def test_get_default_search_space():
    space = get_default_search_space()
    assert 'learning_rate' in space
    assert 'lora_rank' in space
    assert 'batch_size' in space


def test_create_objective():
    objective = create_objective(
        train_dataset_path="dummy.jsonl",
        eval_dataset_path="dummy.jsonl",
        base_model="gpt-oss:20b",
        output_dir="/tmp/test"
    )
    assert callable(objective)
