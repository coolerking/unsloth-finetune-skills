from pathlib import Path
from skills.shared.paths import get_run_paths, ensure_run_dirs

def test_get_run_paths():
    paths = get_run_paths("20260323_120000_test1")
    assert paths["base"].name == "20260323_120000_test1"
    assert paths["dataset"].name == "00_dataset"
    assert paths["training"].name == "01_training"

def test_ensure_run_dirs(tmp_path):
    paths = get_run_paths("test_run", base_dir=str(tmp_path))
    ensure_run_dirs(paths)
    assert paths["dataset"].exists()
    assert paths["training"].exists()
    assert paths["logs"].exists()
