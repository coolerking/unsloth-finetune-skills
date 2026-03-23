"""Configuration utilities."""
from pathlib import Path
from typing import Any, Dict, Optional
import json


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary containing the configuration.

    Raises:
        ConfigError: If the file is not found or contains invalid JSON.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Configuration file not found: {config_path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in configuration file: {config_path} - {e}") from e


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        "output_dir": "/workspace/outputs",
        "log_level": "INFO",
        "max_workers": 4,
    }
