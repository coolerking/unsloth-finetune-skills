"""Integration test for the full fine-tuning workflow."""
import pytest
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from skills.shared.run_id import generate_run_id
from skills.shared.paths import get_run_paths, ensure_run_dirs
from skills.unsloth_dataset_creator.pdf_processor import extract_text_from_pdf, get_all_pdf_files
from skills.unsloth_dataset_creator.chunker import chunk_text, count_tokens, split_by_sections, split_by_tokens
from skills.unsloth_dataset_creator.qa_generator import validate_qa, parse_qa_response, create_qa_generation_prompt
from skills.unsloth_trainer.optuna_config import get_default_search_space, sample_params


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_pdf_structure(tmp_path):
    """Create a sample PDF directory structure for testing."""
    category_dir = tmp_path / "category_a"
    category_dir.mkdir()
    pdf_file = category_dir / "test.pdf"
    pdf_file.write_text("dummy pdf content")
    return tmp_path, category_dir, pdf_file


@pytest.fixture
def sample_doc_info():
    """Return sample document info for chunking tests."""
    return {
        'filename': 'test_doc.pdf',
        'category': 'regulations',
        'text': '第1条 This is the first section of text.\n第2条 This is the second section with more content.'
    }


@pytest.fixture
def sample_chunk():
    """Return sample chunk for Q&A generation tests."""
    return {
        'filename': 'test.pdf',
        'category': 'regulations',
        'text': 'This is a sample regulation text for testing purposes. ' * 20,
        'chunk_id': 'test_chunk_001',
        'token_count': 100
    }


@pytest.fixture
def valid_qa():
    """Return a valid Q&A dictionary."""
    return {
        'question': 'What is the regulation about?',
        'answer': 'This is a detailed answer that explains the regulation in depth with sufficient length to pass validation.',
        'thinking': 'The thinking process involves analyzing the text and extracting key information.'
    }


# =============================================================================
# Run ID and Path Tests
# =============================================================================

def test_run_id_generation():
    """Test that run_id is generated in correct format."""
    run_id = generate_run_id()
    assert len(run_id) > 0
    assert '_' in run_id
    # Verify format: YYYYMMDD_HHMMSS_suffix
    parts = run_id.split('_')
    assert len(parts) == 3  # YYYYMMDD, HHMMSS, suffix
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 6  # HHMMSS
    assert len(parts[2]) == 4  # suffix
    # Verify timestamp is valid
    timestamp = parts[0] + parts[1]
    assert len(timestamp) == 14
    assert timestamp.isdigit()


def test_run_id_uniqueness():
    """Test that generated run_ids are unique."""
    run_ids = [generate_run_id() for _ in range(10)]
    assert len(set(run_ids)) == len(run_ids)


def test_run_paths_creation(tmp_path):
    """Test that run paths are created correctly."""
    paths = get_run_paths("test_run_001", base_dir=str(tmp_path))
    ensure_run_dirs(paths)
    assert paths["dataset"].exists()
    assert paths["training"].exists()
    assert paths["evaluation"].exists()
    assert paths["logs"].exists()
    assert paths["metadata"].name == "metadata.json"


def test_run_paths_structure():
    """Test that run paths have correct structure."""
    paths = get_run_paths("test_run_001", base_dir="/tmp/test")
    assert paths["base"] == Path("/tmp/test/test_run_001")
    assert paths["dataset"] == Path("/tmp/test/test_run_001/00_dataset")
    assert paths["training"] == Path("/tmp/test/test_run_001/01_training")
    assert paths["evaluation"] == Path("/tmp/test/test_run_001/02_evaluation")
    assert paths["logs"] == Path("/tmp/test/test_run_001/logs")


def test_full_run_workflow(tmp_path):
    """Test complete workflow: run_id -> paths -> directory creation."""
    # Generate run_id
    run_id = generate_run_id()
    assert run_id

    # Get paths
    paths = get_run_paths(run_id, base_dir=str(tmp_path))
    assert paths["base"].name == run_id

    # Create directories
    ensure_run_dirs(paths)
    assert paths["base"].exists()

    # Verify all subdirectories exist
    for key in ["dataset", "training", "evaluation", "logs"]:
        assert paths[key].exists()
        assert paths[key].is_dir()


# =============================================================================
# PDF Processor Tests
# =============================================================================

def test_get_all_pdf_files(sample_pdf_structure):
    """Test finding all PDF files in directory."""
    tmp_path, category_dir, pdf_file = sample_pdf_structure
    pdf_files = get_all_pdf_files(tmp_path)
    assert len(pdf_files) == 1
    assert pdf_files[0].name == "test.pdf"


def test_get_all_pdf_files_excludes_checkpoints(sample_pdf_structure):
    """Test that checkpoint directories are excluded."""
    tmp_path, category_dir, pdf_file = sample_pdf_structure
    # Create checkpoint directory with PDF
    checkpoint_dir = category_dir / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_pdf = checkpoint_dir / "checkpoint.pdf"
    checkpoint_pdf.write_text("checkpoint content")

    pdf_files = get_all_pdf_files(tmp_path)
    assert len(pdf_files) == 1
    assert pdf_files[0].name == "test.pdf"


@patch('skills.unsloth_dataset_creator.pdf_processor.extract_text')
def test_extract_text_from_pdf_success(mock_extract_text, sample_pdf_structure):
    """Test PDF text extraction with mock."""
    tmp_path, category_dir, pdf_file = sample_pdf_structure
    mock_extract_text.return_value = "Extracted PDF text content"

    result = extract_text_from_pdf(pdf_file)

    assert result is not None
    assert result['filename'] == "test.pdf"
    assert result['category'] == "category_a"
    assert result['text'] == "Extracted PDF text content"
    assert result['char_count'] == len("Extracted PDF text content")
    assert result['path'] == str(pdf_file)


@patch('skills.unsloth_dataset_creator.pdf_processor.extract_text')
def test_extract_text_from_pdf_error(mock_extract_text, sample_pdf_structure):
    """Test PDF extraction handles errors gracefully."""
    tmp_path, category_dir, pdf_file = sample_pdf_structure
    mock_extract_text.side_effect = Exception("PDF read error")

    result = extract_text_from_pdf(pdf_file)

    assert result is None


# =============================================================================
# Chunker Tests
# =============================================================================

def test_count_tokens():
    """Test token counting function."""
    text = "Hello world"
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0


def test_split_by_sections():
    """Test section-based text splitting."""
    text = "第1条 First section\nSome content\n第2条 Second section\nMore content"
    sections = split_by_sections(text)
    assert len(sections) == 2
    assert "第1条" in sections[0]
    assert "第2条" in sections[1]


def test_split_by_sections_custom_keywords():
    """Test section splitting with custom keywords."""
    text = "Section A\nContent\nSection B\nMore content"
    sections = split_by_sections(text, section_keywords=["Section"])
    assert len(sections) == 2


def test_split_by_tokens():
    """Test token-based text splitting."""
    text = "This is a test sentence with multiple words. " * 50
    chunks = split_by_tokens(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    # Verify overlap works
    if len(chunks) > 1:
        assert len(chunks[0]) > len(chunks[1]) * 0.5  # Some overlap expected


def test_chunk_text(sample_doc_info):
    """Test full chunking pipeline."""
    chunks = chunk_text(sample_doc_info, chunk_size=100, chunk_overlap=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    for chunk in chunks:
        assert 'text' in chunk
        assert 'filename' in chunk
        assert 'category' in chunk
        assert 'chunk_id' in chunk
        assert 'token_count' in chunk
        assert chunk['filename'] == sample_doc_info['filename']
        assert chunk['category'] == sample_doc_info['category']
        assert chunk['token_count'] > 0


def test_chunk_text_large_section():
    """Test chunking when section exceeds chunk_size."""
    doc_info = {
        'filename': 'large_doc.pdf',
        'category': 'test',
        'text': '第1条 ' + 'word ' * 1000  # Large section
    }
    chunks = chunk_text(doc_info, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


# =============================================================================
# Q&A Generator Tests
# =============================================================================

def test_validate_qa_valid(valid_qa):
    """Test validation of valid Q&A."""
    assert validate_qa(valid_qa) is True


def test_validate_qa_short_answer(valid_qa):
    """Test validation fails for short answer."""
    qa = valid_qa.copy()
    qa['answer'] = 'Short'
    assert validate_qa(qa, min_answer_length=50) is False


def test_validate_qa_empty_question(valid_qa):
    """Test validation fails for empty question."""
    qa = valid_qa.copy()
    qa['question'] = ''
    assert validate_qa(qa) is False


def test_validate_qa_empty_answer(valid_qa):
    """Test validation fails for empty answer."""
    qa = valid_qa.copy()
    qa['answer'] = ''
    assert validate_qa(qa) is False


def test_validate_qa_empty_thinking(valid_qa):
    """Test validation fails for empty thinking."""
    qa = valid_qa.copy()
    qa['thinking'] = ''
    assert validate_qa(qa) is False


def test_parse_qa_response_json():
    """Test parsing JSON response."""
    response = '{"question": "Q?", "answer": "A" * 100, "thinking": "T"}'
    result = parse_qa_response(response.replace('"A" * 100', '"' + 'A' * 100 + '"'))
    assert result is not None


def test_parse_qa_response_with_code_block():
    """Test parsing response with markdown code block."""
    response = '''```json
{"question": "Q?", "answer": "Answer text here", "thinking": "Thinking process"}
```'''
    result = parse_qa_response(response)
    assert result is not None
    assert result['question'] == "Q?"


def test_parse_qa_response_invalid_json():
    """Test parsing invalid JSON response."""
    response = "Not valid JSON"
    result = parse_qa_response(response)
    assert result is None


def test_create_qa_generation_prompt(sample_chunk):
    """Test prompt creation for Q&A generation."""
    prompt = create_qa_generation_prompt(sample_chunk, "事実確認型", 1)

    assert sample_chunk['filename'] in prompt
    assert sample_chunk['category'] in prompt
    assert sample_chunk['text'][:100] in prompt
    assert "事実確認型" in prompt
    assert "JSON" in prompt


def test_create_qa_generation_prompt_truncates_long_text(sample_chunk):
    """Test that long text is truncated in prompt."""
    sample_chunk['text'] = 'word ' * 1000  # Long text
    prompt = create_qa_generation_prompt(sample_chunk, "手続き型", 1)
    assert '...' in prompt or len(prompt) < len(sample_chunk['text'])


# =============================================================================
# Optuna Config Tests
# =============================================================================

def test_get_default_search_space():
    """Test default search space structure."""
    space = get_default_search_space()

    required_keys = ['learning_rate', 'lora_rank', 'lora_alpha', 'lora_dropout', 'batch_size', 'num_epochs']
    for key in required_keys:
        assert key in space

    # Verify structure of each parameter
    assert space['learning_rate']['type'] == 'float'
    assert space['lora_rank']['type'] == 'categorical'
    assert 'choices' in space['lora_rank']


def test_sample_params():
    """Test parameter sampling from search space."""
    space = get_default_search_space()

    # Create a mock trial
    mock_trial = MagicMock()
    mock_trial.suggest_float.return_value = 1e-4
    mock_trial.suggest_int.return_value = 3
    mock_trial.suggest_categorical.side_effect = [16, 32, 2]

    params = sample_params(mock_trial, space)

    assert 'learning_rate' in params
    assert 'lora_rank' in params
    assert 'lora_alpha' in params
    assert 'lora_dropout' in params
    assert 'batch_size' in params
    assert 'num_epochs' in params


# =============================================================================
# Integration Workflow Tests
# =============================================================================

def test_end_to_end_chunking_workflow():
    """Test full workflow from doc_info to chunks."""
    doc_info = {
        'filename': 'integration_test.pdf',
        'category': 'hr_policies',
        'text': '''第1条 Employee Conduct
All employees must follow company policies and maintain professional behavior.
第2条 Leave Policy
Employees are entitled to annual leave based on their years of service.'''
    }

    # Chunk the document
    chunks = chunk_text(doc_info, chunk_size=100, chunk_overlap=20)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk['filename'] == 'integration_test.pdf'
        assert chunk['category'] == 'hr_policies'
        assert chunk['token_count'] <= 100 + 20  # Allow some margin


def test_run_id_to_chunk_workflow(tmp_path):
    """Test workflow from run_id generation through directory creation to chunk storage."""
    # Generate run_id
    run_id = generate_run_id()

    # Create paths
    paths = get_run_paths(run_id, base_dir=str(tmp_path))
    ensure_run_dirs(paths)

    # Create sample chunks
    chunks = [
        {
            'text': 'Sample chunk 1',
            'filename': 'doc1.pdf',
            'category': 'cat1',
            'chunk_id': 'chunk_001',
            'token_count': 10
        },
        {
            'text': 'Sample chunk 2',
            'filename': 'doc2.pdf',
            'category': 'cat2',
            'chunk_id': 'chunk_002',
            'token_count': 15
        }
    ]

    # Save chunks to dataset directory
    chunks_file = paths['dataset'] / 'chunks.json'
    with open(chunks_file, 'w') as f:
        json.dump(chunks, f)

    # Verify file was created and content is correct
    assert chunks_file.exists()
    with open(chunks_file, 'r') as f:
        loaded_chunks = json.load(f)
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0]['chunk_id'] == 'chunk_001'
