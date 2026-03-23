"""Optuna hyperparameter search configuration."""
from typing import Dict, Callable, Any


def get_default_search_space() -> Dict[str, Any]:
    """Get default hyperparameter search space."""
    return {
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
        'lora_rank': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
        'lora_alpha': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
        'lora_dropout': {'type': 'float', 'low': 0.0, 'high': 0.1},
        'batch_size': {'type': 'categorical', 'choices': [1, 2, 4]},
        'num_epochs': {'type': 'int', 'low': 1, 'high': 5},
    }


def sample_params(trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters from search space."""
    params = {}
    for name, config in search_space.items():
        if config['type'] == 'float':
            params[name] = trial.suggest_float(
                name, config['low'], config['high'], log=config.get('log', False)
            )
        elif config['type'] == 'int':
            params[name] = trial.suggest_int(name, config['low'], config['high'])
        elif config['type'] == 'categorical':
            params[name] = trial.suggest_categorical(name, config['choices'])
    return params


def create_objective(
    train_dataset_path: str,
    eval_dataset_path: str,
    base_model: str,
    output_dir: str
) -> Callable:
    """Create Optuna objective function."""
    search_space = get_default_search_space()

    def objective(trial):
        params = sample_params(trial, search_space)
        # Placeholder - actual training in training_loop.py
        return 0.5  # Return eval_loss

    return objective
