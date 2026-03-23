"""Integration tests for Optuna hyperparameter optimization."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


# Mock unsloth, optuna, and related imports before importing the module
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
        import sys
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
        # Create trial mocks with proper state attribute for JSON serialization
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
def mock_train_dataset(tmp_path):
    """Create a mock training dataset file."""
    dataset_file = tmp_path / "train.jsonl"
    test_data = [
        {"instruction": "Test 1", "output": "Output 1", "thinking": "Think 1"},
        {"instruction": "Test 2", "output": "Output 2"},
    ]
    with open(dataset_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    return str(dataset_file)


@pytest.fixture
def mock_eval_dataset(tmp_path):
    """Create a mock evaluation dataset file."""
    dataset_file = tmp_path / "eval.jsonl"
    test_data = [
        {"instruction": "Eval 1", "output": "Eval Output 1"},
    ]
    with open(dataset_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    return str(dataset_file)


class TestOptunaIntegration:
    """Tests for Optuna integration with training."""

    def test_create_objective_calls_train_with_unsloth(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test that create_objective properly calls train_with_unsloth."""
        from unsloth_trainer import optuna_config

        output_dir = str(tmp_path / "output")

        # Mock train_with_unsloth at the module level where it's imported
        with patch.object(optuna_config, 'train_with_unsloth') as mock_train:
            mock_train.return_value = {'eval_loss': 0.3, 'train_loss': 0.5}

            # Create objective
            objective = optuna_config.create_objective(
                train_dataset_path=mock_train_dataset,
                eval_dataset_path=mock_eval_dataset,
                base_model="unsloth/llama-3-8b",
                output_dir=output_dir
            )

            # Create mock trial
            mock_trial = MagicMock()
            mock_trial.number = 0
            mock_trial.suggest_float.side_effect = [5e-5, 0.05]  # lr, dropout
            mock_trial.suggest_int.return_value = 3  # num_epochs
            mock_trial.suggest_categorical.side_effect = [16, 32, 2]  # rank, alpha, batch

            # Call objective
            result = objective(mock_trial)

            # Verify train_with_unsloth was called
            assert mock_train.called
            call_kwargs = mock_train.call_args[1]
            assert call_kwargs['train_dataset_path'] == mock_train_dataset
            assert call_kwargs['eval_dataset_path'] == mock_eval_dataset
            assert call_kwargs['base_model'] == "unsloth/llama-3-8b"
            assert 'params' in call_kwargs

            # Verify result is eval_loss
            assert result == 0.3

    def test_run_optuna_study(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test run_optuna_study function."""
        from unsloth_trainer import optuna_config

        mock_optuna, mock_study = mock_unsloth_and_optuna
        output_dir = str(tmp_path / "output")

        with patch.object(optuna_config, 'train_with_unsloth') as mock_train:
            mock_train.return_value = {'eval_loss': 0.3, 'train_loss': 0.5}

            result = optuna_config.run_optuna_study(
                train_dataset_path=mock_train_dataset,
                eval_dataset_path=mock_eval_dataset,
                base_model="unsloth/llama-3-8b",
                output_dir=output_dir,
                n_trials=3
            )

            # Verify study was created
            mock_optuna.create_study.assert_called_once()

            # Verify optimize was called with correct n_trials
            mock_study.optimize.assert_called_once()
            call_args = mock_study.optimize.call_args
            # n_trials is passed as keyword argument
            assert call_args[1].get('n_trials') == 3

            # Verify results
            assert result['best_params'] == mock_study.best_params
            assert result['best_value'] == mock_study.best_value
            assert result['best_trial_number'] == 2
            assert result['n_trials_completed'] == 3

            # Verify study file was saved
            study_file = Path(result['study_file'])
            assert study_file.exists()

    def test_fine_tune_entry_point(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test the fine_tune entry point function."""
        from unsloth_trainer import optuna_config

        mock_optuna, mock_study = mock_unsloth_and_optuna
        output_dir = str(tmp_path / "output")

        with patch.object(optuna_config, 'train_with_unsloth') as mock_train:
            mock_train.return_value = {'eval_loss': 0.25, 'train_loss': 0.4}

            # Import here after mock is set up
            from unsloth_trainer import fine_tune

            result = fine_tune(
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                output_dir=output_dir,
                base_model="unsloth/llama-3-8b",
                optuna_config={"n_trials": 3}
            )

            # Verify success
            assert result['status'] == 'success'
            assert 'best_model_path' in result
            assert 'best_params' in result
            assert result['best_eval_loss'] == 0.25

            # Verify best_model directory was created
            best_model_path = Path(result['best_model_path'])
            assert best_model_path.exists()

    def test_fine_tune_default_config(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test fine_tune with default configuration."""
        from unsloth_trainer import optuna_config

        mock_optuna, mock_study = mock_unsloth_and_optuna
        output_dir = str(tmp_path / "output")

        with patch.object(optuna_config, 'train_with_unsloth') as mock_train:
            mock_train.return_value = {'eval_loss': 0.25, 'train_loss': 0.4}

            from unsloth_trainer import fine_tune

            # Call without optuna_config (should use defaults)
            result = fine_tune(
                train_dataset=mock_train_dataset,
                eval_dataset=mock_eval_dataset,
                output_dir=output_dir,
                base_model="unsloth/llama-3-8b"
            )

            assert result['status'] == 'success'
            # Default n_trials is 20
            mock_study.optimize.assert_called_once()

    def test_fine_tune_missing_dataset(self, tmp_path, mock_unsloth_and_optuna):
        """Test fine_tune with missing dataset files."""
        from unsloth_trainer import fine_tune

        output_dir = str(tmp_path / "output")

        with pytest.raises(FileNotFoundError):
            fine_tune(
                train_dataset="/nonexistent/train.jsonl",
                eval_dataset="/nonexistent/eval.jsonl",
                output_dir=output_dir
            )

    def test_fine_tune_error_handling(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test fine_tune error handling."""
        from unsloth_trainer import optuna_config

        mock_optuna, mock_study = mock_unsloth_and_optuna
        output_dir = str(tmp_path / "output")

        # Make the study raise an exception during optimization
        mock_study.optimize.side_effect = Exception("Training failed")

        from unsloth_trainer import fine_tune

        result = fine_tune(
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            output_dir=output_dir,
            base_model="unsloth/llama-3-8b",
            optuna_config={"n_trials": 1}
        )

        assert result['status'] == 'error'
        assert 'error' in result
        assert 'Training failed' in result['error']

    def test_trial_specific_output_directory(
        self, tmp_path, mock_train_dataset, mock_eval_dataset, mock_unsloth_and_optuna
    ):
        """Test that each trial gets its own output directory."""
        from unsloth_trainer import optuna_config

        output_dir = str(tmp_path / "output")

        with patch.object(optuna_config, 'train_with_unsloth') as mock_train:
            mock_train.return_value = {'eval_loss': 0.3, 'train_loss': 0.5}

            objective = optuna_config.create_objective(
                train_dataset_path=mock_train_dataset,
                eval_dataset_path=mock_eval_dataset,
                base_model="unsloth/llama-3-8b",
                output_dir=output_dir
            )

            # Test trial 5
            mock_trial = MagicMock()
            mock_trial.number = 5
            mock_trial.suggest_float.side_effect = [5e-5, 0.05]
            mock_trial.suggest_int.return_value = 3
            mock_trial.suggest_categorical.side_effect = [16, 32, 2]

            objective(mock_trial)

            # Verify trial-specific directory was used
            call_kwargs = mock_train.call_args[1]
            assert "trials/trial_5" in call_kwargs['output_dir']
