"""Path management utilities."""
from pathlib import Path
from typing import Dict

def get_run_paths(run_id: str, base_dir: str = "/workspace/outputs") -> Dict[str, Path]:
    """Get standardized paths for a run."""
    base = Path(base_dir) / run_id
    return {
        "base": base,
        "dataset": base / "00_dataset",
        "training": base / "01_training",
        "evaluation": base / "02_evaluation",
        "logs": base / "logs",
        "metadata": base / "metadata.json"
    }

def ensure_run_dirs(paths: Dict[str, Path]) -> None:
    """Create all run directories."""
    for key, path in paths.items():
        if key != "metadata":
            path.mkdir(parents=True, exist_ok=True)
