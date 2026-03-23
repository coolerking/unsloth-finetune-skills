"""Configuration utilities."""
from pathlib import Path
from typing import Any, Dict, Optional
import json


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


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
