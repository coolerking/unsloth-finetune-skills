"""Tests for shared/config.py."""
import json
import pytest
from pathlib import Path
from shared.config import load_config, save_config, get_default_config, ConfigError


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_success(self, tmp_path):
        """Test successful loading of a valid JSON config file."""
        config_path = tmp_path / "config.json"
        expected_config = {"key1": "value1", "key2": 42}

        with open(config_path, 'w') as f:
            json.dump(expected_config, f)

        result = load_config(config_path)
        assert result == expected_config

    def test_load_config_file_not_found(self, tmp_path):
        """Test that ConfigError is raised when file does not exist."""
        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "Configuration file not found" in str(exc_info.value)
        assert str(config_path) in str(exc_info.value)

    def test_load_config_invalid_json(self, tmp_path):
        """Test that ConfigError is raised when file contains invalid JSON."""
        config_path = tmp_path / "invalid.json"

        with open(config_path, 'w') as f:
            f.write("{invalid json content}")

        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)

        assert "Invalid JSON" in str(exc_info.value)
        assert str(config_path) in str(exc_info.value)


class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_config(self, tmp_path):
        """Test saving configuration to a JSON file."""
        config_path = tmp_path / "subdir" / "config.json"
        config = {"key1": "value1", "key2": 42}

        save_config(config, config_path)

        assert config_path.exists()
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config == config


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_get_default_config(self):
        """Test getting default configuration values."""
        config = get_default_config()

        assert "output_dir" in config
        assert "log_level" in config
        assert "max_workers" in config
        assert config["output_dir"] == "/workspace/outputs"
        assert config["log_level"] == "INFO"
        assert config["max_workers"] == 4
