"""Optuna hyperparameter search configuration."""
from typing import Dict, Callable, Any
from pathlib import Path

from .training_loop import train_with_unsloth


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
    """Create Optuna objective function.

    Args:
        train_dataset_path: Path to the training dataset JSONL file.
        eval_dataset_path: Path to the evaluation dataset JSONL file.
        base_model: Name or path of the base model to fine-tune.
        output_dir: Directory to save trial outputs.

    Returns:
        Objective function for Optuna optimization.
    """
    search_space = get_default_search_space()
    output_path = Path(output_dir)

    def objective(trial):
        """Optuna objective function that trains and returns eval_loss."""
        params = sample_params(trial, search_space)

        # Create trial-specific output directory
        trial_output_dir = output_path / "trials" / f"trial_{trial.number}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        # Run training with unsloth
        result = train_with_unsloth(
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
            output_dir=str(trial_output_dir),
            base_model=base_model,
            params=params
        )

        # Return eval_loss for optimization (lower is better)
        return result['eval_loss']

    return objective


def run_optuna_study(
    train_dataset_path: str,
    eval_dataset_path: str,
    base_model: str,
    output_dir: str,
    n_trials: int = 20,
    study_name: str = "unsloth_optimization"
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization study.

    Args:
        train_dataset_path: Path to the training dataset JSONL file.
        eval_dataset_path: Path to the evaluation dataset JSONL file.
        base_model: Name or path of the base model to fine-tune.
        output_dir: Directory to save outputs.
        n_trials: Number of Optuna trials to run.
        study_name: Name for the Optuna study.

    Returns:
        Dictionary containing study results:
        - best_params: Best hyperparameters found
        - best_value: Best eval_loss achieved
        - best_trial_number: Trial number of best result
        - n_trials_completed: Number of trials completed
        - output_dir: Path to output directory
    """
    import optuna
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create objective function
    objective = create_objective(
        train_dataset_path=train_dataset_path,
        eval_dataset_path=eval_dataset_path,
        base_model=base_model,
        output_dir=output_dir
    )

    # Create and run study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials)

    # Save study results
    study_results = {
        "study_name": study_name,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": len(study.trials),
        "trials": [
            {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
    }

    study_file = output_path / "optuna_study.json"
    with open(study_file, 'w') as f:
        json.dump(study_results, f, indent=2)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": len(study.trials),
        "output_dir": output_dir,
        "study_file": str(study_file)
    }
