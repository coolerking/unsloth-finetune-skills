"""Integration test for the full fine-tuning workflow."""
import pytest
from pathlib import Path
from skills.shared.run_id import generate_run_id
from skills.shared.paths import get_run_paths, ensure_run_dirs

def test_run_id_generation():
    run_id = generate_run_id()
    assert len(run_id) > 0
    assert '_' in run_id

def test_run_paths_creation(tmp_path):
    paths = get_run_paths("test_run_001", base_dir=str(tmp_path))
    ensure_run_dirs(paths)
    assert paths["dataset"].exists()
    assert paths["training"].exists()
    assert paths["evaluation"].exists()
    assert paths["logs"].exists()

def test_pdf_processor_import():
    from skills.unsloth_dataset_creator.pdf_processor import extract_text_from_pdf
    assert callable(extract_text_from_pdf)

def test_chunker_import():
    from skills.unsloth_dataset_creator.chunker import chunk_text
    assert callable(chunk_text)

def test_qa_generator_import():
    from skills.unsloth_dataset_creator.qa_generator import validate_qa
    assert callable(validate_qa)

def test_optuna_config_import():
    from skills.unsloth_trainer.optuna_config import get_default_search_space
    assert callable(get_default_search_space)
