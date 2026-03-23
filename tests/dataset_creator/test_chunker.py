import pytest
from skills.unsloth_dataset_creator.chunker import (
    count_tokens,
    split_by_sections,
    split_by_tokens,
    chunk_text
)


def test_count_tokens():
    text = "Hello world"
    count = count_tokens(text)
    assert count > 0
    assert isinstance(count, int)


def test_split_by_sections():
    text = """第一章
This is section 1.
Some content here.

第二章
This is section 2.
More content."""
    sections = split_by_sections(text, section_keywords=["章"])
    assert len(sections) == 2
    assert "第一章" in sections[0]
    assert "第二章" in sections[1]


def test_split_by_tokens():
    text = "This is a test sentence. " * 50
    chunks = split_by_tokens(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1


def test_chunk_text():
    doc_info = {
        'text': '第一章\nContent here.\n\n第二章\nMore content here.',
        'filename': 'test.pdf',
        'category': 'test'
    }
    chunks = chunk_text(doc_info, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    assert 'chunk_id' in chunks[0]
    assert 'token_count' in chunks[0]


def test_split_by_tokens_empty_text():
    """Test split_by_tokens handles empty text."""
    assert split_by_tokens("", chunk_size=20, overlap=5) == []
    assert split_by_tokens("   ", chunk_size=20, overlap=5) == []
    assert split_by_tokens("\n\n", chunk_size=20, overlap=5) == []


def test_chunk_text_empty_text():
    """Test chunk_text handles empty text gracefully."""
    doc_info_empty = {
        'text': '',
        'filename': 'test.pdf',
        'category': 'test'
    }
    assert chunk_text(doc_info_empty) == []

    doc_info_whitespace = {
        'text': '   \n\n  ',
        'filename': 'test.pdf',
        'category': 'test'
    }
    assert chunk_text(doc_info_whitespace) == []

    doc_info_missing_text = {
        'filename': 'test.pdf',
        'category': 'test'
    }
    assert chunk_text(doc_info_missing_text) == []


def test_chunk_text_very_short_text():
    """Test chunk_text handles very short text."""
    doc_info = {
        'text': 'Hi',
        'filename': 'test.pdf',
        'category': 'test'
    }
    chunks = chunk_text(doc_info, chunk_size=50, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0]['text'] == 'Hi'


def test_split_by_tokens_invalid_overlap():
    """Test that chunk_size <= overlap raises ValueError."""
    text = "This is a test sentence. " * 50
    with pytest.raises(ValueError, match="chunk_size .* must be greater than overlap"):
        split_by_tokens(text, chunk_size=10, overlap=10)
    with pytest.raises(ValueError, match="chunk_size .* must be greater than overlap"):
        split_by_tokens(text, chunk_size=5, overlap=10)
