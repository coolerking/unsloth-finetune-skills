# tests/dataset_creator/test_pdf_processor.py
from pathlib import Path
from unittest.mock import patch, MagicMock
from skills.unsloth_dataset_creator.pdf_processor import extract_text_from_pdf, get_all_pdf_files

def test_extract_text_from_pdf_success():
    mock_doc = {
        'path': '/test/doc.pdf',
        'filename': 'doc.pdf',
        'category': 'hr',
        'text': 'Sample text content',
        'char_count': 19
    }
    with patch('skills.unsloth_dataset_creator.pdf_processor.extract_text') as mock_extract:
        mock_extract.return_value = 'Sample text content'
        result = extract_text_from_pdf(Path('/test/doc.pdf'))
        assert result['filename'] == 'doc.pdf'
        assert result['text'] == 'Sample text content'
        assert result['char_count'] == 19

def test_get_all_pdf_files(tmp_path):
    (tmp_path / 'test1.pdf').touch()
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'test2.pdf').touch()
    (tmp_path / 'notpdf.txt').touch()
    files = get_all_pdf_files(tmp_path)
    assert len(files) == 2
    assert all(f.suffix == '.pdf' for f in files)
