"""Tests for training_loop.py."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


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
    }):
        yield


class TestLoadJsonlDataset:
    """Tests for load_jsonl_dataset function."""

    def test_load_valid_jsonl(self, tmp_path, mock_unsloth):
        """Test loading a valid JSONL file."""
        from unsloth_trainer.training_loop import load_jsonl_dataset

        # Create a test JSONL file
        test_file = tmp_path / "test.jsonl"
        test_data = [
            {"instruction": "Test 1", "output": "Output 1", "thinking": "Think 1"},
            {"instruction": "Test 2", "output": "Output 2"},
        ]
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')

        # Mock datasets.load_dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=2)

        with patch('unsloth_trainer.training_loop.load_dataset') as mock_load:
            mock_load.return_value = mock_dataset
            result = load_jsonl_dataset(str(test_file))
            mock_load.assert_called_once_with('json', data_files=str(test_file))
            assert result == mock_dataset

    def test_file_not_found(self, mock_unsloth):
        """Test handling of non-existent file."""
        from unsloth_trainer.training_loop import load_jsonl_dataset

        with pytest.raises(FileNotFoundError):
            load_jsonl_dataset("/nonexistent/path.jsonl")


class TestFormatInstruction:
    """Tests for format_instruction function."""

    def test_format_with_thinking(self, mock_unsloth):
        """Test formatting with thinking field."""
        from unsloth_trainer.training_loop import format_instruction

        example = {
            "instruction": "What is AI?",
            "output": "AI is artificial intelligence.",
            "thinking": "I need to define AI."
        }

        result = format_instruction(example)

        assert "<|im_start|>user" in result
        assert "What is AI?" in result
        assert "<|im_start|>assistant" in result
        assert "<thinking>I need to define AI.</thinking>" in result
        assert "AI is artificial intelligence." in result
        assert "<|im_end|>" in result

    def test_format_without_thinking(self, mock_unsloth):
        """Test formatting without thinking field."""
        from unsloth_trainer.training_loop import format_instruction

        example = {
            "instruction": "What is ML?",
            "output": "ML is machine learning."
        }

        result = format_instruction(example)

        assert "<|im_start|>user" in result
        assert "What is ML?" in result
        assert "<|im_start|>assistant" in result
        assert "ML is machine learning." in result
        assert "<thinking>" not in result
        assert "<|im_end|>" in result

    def test_format_missing_instruction(self, mock_unsloth):
        """Test formatting with missing instruction field."""
        from unsloth_trainer.training_loop import format_instruction

        example = {"output": "Just output."}

        with pytest.raises(KeyError):
            format_instruction(example)

    def test_format_missing_output(self, mock_unsloth):
        """Test formatting with missing output field."""
        from unsloth_trainer.training_loop import format_instruction

        example = {"instruction": "Just instruction."}

        with pytest.raises(KeyError):
            format_instruction(example)


class TestTrainWithUnsloth:
    """Tests for train_with_unsloth function."""

    @pytest.fixture
    def mock_train_dataset(self):
        """Create a mock training dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        dataset.map = MagicMock(return_value=dataset)
        return dataset

    @pytest.fixture
    def mock_eval_dataset(self):
        """Create a mock evaluation dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=20)
        dataset.map = MagicMock(return_value=dataset)
        return dataset

    @pytest.fixture
    def default_params(self):
        """Default training parameters."""
        return {
            'learning_rate': 2e-4,
            'lora_rank': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'batch_size': 2,
            'num_epochs': 1
        }

    def test_train_with_unsloth_success(
        self, tmp_path, mock_unsloth, mock_train_dataset, mock_eval_dataset, default_params
    ):
        """Test successful training."""
        from unsloth_trainer.training_loop import train_with_unsloth

        # Mock unsloth FastLanguageModel
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|im_end|>"

        with patch('unsloth_trainer.training_loop.FastLanguageModel') as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_flm.get_peft_model.return_value = (mock_model, mock_tokenizer)

            # Mock SFTTrainer
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock(
                training_loss=0.5
            )
            mock_trainer.evaluate.return_value = {'eval_loss': 0.3}

            with patch('unsloth_trainer.training_loop.SFTTrainer', return_value=mock_trainer):
                with patch('unsloth_trainer.training_loop.load_jsonl_dataset') as mock_load:
                    mock_load.side_effect = [mock_train_dataset, mock_eval_dataset]

                    output_dir = str(tmp_path / "output")
                    result = train_with_unsloth(
                        train_dataset_path="train.jsonl",
                        eval_dataset_path="eval.jsonl",
                        output_dir=output_dir,
                        base_model="unsloth/llama-3-8b",
                        params=default_params
                    )

                    # Verify results
                    assert 'eval_loss' in result
                    assert 'train_loss' in result
                    assert result['eval_loss'] == 0.3
                    assert result['train_loss'] == 0.5

                    # Verify model was saved
                    mock_model.save_pretrained.assert_called_once()
                    mock_tokenizer.save_pretrained.assert_called_once()

    def test_train_with_unsloth_lora_config(
        self, tmp_path, mock_unsloth, mock_train_dataset, mock_eval_dataset, default_params
    ):
        """Test that LoRA configuration is passed correctly."""
        from unsloth_trainer.training_loop import train_with_unsloth

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|im_end|>"

        with patch('unsloth_trainer.training_loop.FastLanguageModel') as mock_flm:
            mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
            mock_flm.get_peft_model.return_value = (mock_model, mock_tokenizer)

            mock_trainer = MagicMock()
            mock_trainer.train.return_value = MagicMock(
                training_loss=0.5,
                metrics={'eval_loss': 0.3}
            )

            with patch('unsloth_trainer.training_loop.SFTTrainer', return_value=mock_trainer):
                with patch('unsloth_trainer.training_loop.load_jsonl_dataset') as mock_load:
                    mock_load.side_effect = [mock_train_dataset, mock_eval_dataset]

                    output_dir = str(tmp_path / "output")
                    train_with_unsloth(
                        train_dataset_path="train.jsonl",
                        eval_dataset_path="eval.jsonl",
                        output_dir=output_dir,
                        base_model="unsloth/llama-3-8b",
                        params=default_params
                    )

                    # Verify LoRA config
                    call_kwargs = mock_flm.get_peft_model.call_args[1]
                    assert call_kwargs['r'] == default_params['lora_rank']
                    assert call_kwargs['lora_alpha'] == default_params['lora_alpha']
                    assert call_kwargs['lora_dropout'] == default_params['lora_dropout']

    def test_train_with_unsloth_missing_params(
        self, tmp_path, mock_unsloth
    ):
        """Test handling of missing required parameters."""
        from unsloth_trainer.training_loop import train_with_unsloth

        incomplete_params = {
            'learning_rate': 2e-4,
            # Missing lora_rank, lora_alpha, etc.
        }

        with pytest.raises(KeyError):
            train_with_unsloth(
                train_dataset_path="train.jsonl",
                eval_dataset_path="eval.jsonl",
                output_dir=str(tmp_path),
                base_model="unsloth/llama-3-8b",
                params=incomplete_params
            )
