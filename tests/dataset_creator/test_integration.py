"""Integration tests for unsloth_dataset_creator full workflow."""
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from skills.unsloth_dataset_creator import (
    create_dataset,
    call_llm,
    chunk_text,
    extract_text_from_pdf,
    get_all_pdf_files,
    create_qa_generation_prompt,
    parse_qa_response,
    validate_qa
)


class TestCallLLM:
    """Tests for call_llm function."""

    def test_call_llm_groq_success(self):
        """Test successful Groq API call."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': '{"question": "Test?", "answer": "Test answer.", "thinking": "Test thinking."}'
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()

        with patch('skills.unsloth_dataset_creator.requests.post', return_value=mock_response) as mock_post:
            result = call_llm(
                provider="groq",
                prompt="Test prompt",
                llm_config={"api_key": "test_key", "model": "llama-3.1-8b-instant"}
            )

            assert result is not None
            assert result['question'] == "Test?"
            assert result['answer'] == "Test answer."
            mock_post.assert_called_once()

    def test_call_llm_unsupported_provider(self):
        """Test unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            call_llm(
                provider="unsupported",
                prompt="Test",
                llm_config={}
            )

    def test_call_llm_api_error(self):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")

        with patch('skills.unsloth_dataset_creator.requests.post', return_value=mock_response):
            result = call_llm(
                provider="groq",
                prompt="Test",
                llm_config={"api_key": "test_key"}
            )
            assert result is None


class TestCreateDataset:
    """Tests for create_dataset function."""

    @pytest.fixture
    def mock_pdf_dir(self, tmp_path):
        """Create a temporary directory with mock PDFs."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create mock PDF files (just empty files for testing)
        (pdf_dir / "doc1.pdf").touch()
        subdir = pdf_dir / "subdir"
        subdir.mkdir()
        (subdir / "doc2.pdf").touch()

        return pdf_dir

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return tmp_path / "output"

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for Q&A generation."""
        return {
            "question": "What is the policy?",
            "answer": "This is a detailed answer that meets the minimum length requirement for validation.",
            "thinking": "I analyzed the document content."
        }

    def test_create_dataset_basic(self, mock_pdf_dir, output_dir, mock_llm_response):
        """Test basic dataset creation workflow."""

        mock_doc_info = {
            'path': str(mock_pdf_dir / "doc1.pdf"),
            'filename': 'doc1.pdf',
            'category': 'pdfs',
            'text': '第一章\nThis is test content for section one.\n\n第二章\nThis is test content for section two with more details.',
            'char_count': 100
        }

        with patch('skills.unsloth_dataset_creator.get_all_pdf_files', return_value=[
            mock_pdf_dir / "doc1.pdf"
        ]):
            with patch('skills.unsloth_dataset_creator.extract_text_from_pdf', return_value=mock_doc_info):
                with patch('skills.unsloth_dataset_creator.call_llm', return_value=mock_llm_response):
                    result = create_dataset(
                        pdf_dir=str(mock_pdf_dir),
                        output_dir=str(output_dir),
                        config={"use_rag": False, "use_thinking": True, "use_tools": False, "target_samples": 2},
                        llm_provider="groq",
                        llm_config={"api_key": "test_key", "model": "llama-3.1-8b-instant"}
                    )

        assert result['status'] == 'success'
        assert 'metadata' in result
        assert result['metadata']['train_count'] > 0
        assert result['metadata']['eval_count'] >= 0

        # Check output files exist
        assert (output_dir / "dataset_train.jsonl").exists()
        assert (output_dir / "dataset_eval.jsonl").exists()
        assert (output_dir / "dataset_metadata.json").exists()

    def test_create_dataset_no_pdfs(self, tmp_path, output_dir):
        """Test handling when no PDFs are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = create_dataset(
            pdf_dir=str(empty_dir),
            output_dir=str(output_dir),
            config={"use_rag": False, "use_thinking": True, "use_tools": False, "target_samples": 10},
            llm_provider="groq",
            llm_config={"api_key": "test_key"}
        )

        assert result['status'] == 'error'
        assert 'No PDF files found' in result['message']

    def test_create_dataset_pdf_extraction_failure(self, mock_pdf_dir, output_dir):
        """Test handling when PDF extraction fails."""

        with patch('skills.unsloth_dataset_creator.get_all_pdf_files', return_value=[
            mock_pdf_dir / "doc1.pdf"
        ]):
            with patch('skills.unsloth_dataset_creator.extract_text_from_pdf', return_value=None):
                result = create_dataset(
                    pdf_dir=str(mock_pdf_dir),
                    output_dir=str(output_dir),
                    config={"use_rag": False, "use_thinking": True, "use_tools": False, "target_samples": 10},
                    llm_provider="groq",
                    llm_config={"api_key": "test_key"}
                )

        assert result['status'] == 'error'
        assert 'Failed to extract text' in result['message']

    def test_create_dataset_output_format(self, mock_pdf_dir, output_dir, mock_llm_response):
        """Test that output JSONL has correct format."""

        mock_doc_info = {
            'path': str(mock_pdf_dir / "doc1.pdf"),
            'filename': 'doc1.pdf',
            'category': 'pdfs',
            'text': '第一章\nTest content here with enough text to create meaningful chunks for processing.',
            'char_count': 100
        }

        with patch('skills.unsloth_dataset_creator.get_all_pdf_files', return_value=[
            mock_pdf_dir / "doc1.pdf"
        ]):
            with patch('skills.unsloth_dataset_creator.extract_text_from_pdf', return_value=mock_doc_info):
                with patch('skills.unsloth_dataset_creator.call_llm', return_value=mock_llm_response):
                    create_dataset(
                        pdf_dir=str(mock_pdf_dir),
                        output_dir=str(output_dir),
                        config={"use_rag": False, "use_thinking": True, "use_tools": False, "target_samples": 1},
                        llm_provider="groq",
                        llm_config={"api_key": "test_key"}
                    )

        # Verify JSONL format (check either train or eval since split may put all in eval for small samples)
        jsonl_file = output_dir / "dataset_train.jsonl"
        with open(jsonl_file, 'r') as f:
            content = f.read().strip()
            if not content:
                # All samples went to eval
                jsonl_file = output_dir / "dataset_eval.jsonl"
                with open(jsonl_file, 'r') as f2:
                    content = f2.read().strip()

        line = json.loads(content.split('\n')[0])
        assert 'instruction' in line
        assert 'output' in line
        assert 'thinking' in line
        assert 'metadata' in line
        assert 'source_file' in line['metadata']
        assert 'category' in line['metadata']
        assert 'chunk_id' in line['metadata']

    def test_create_dataset_train_eval_split(self, mock_pdf_dir, output_dir, mock_llm_response):
        """Test train/eval split is approximately 90/10."""

        mock_doc_info = {
            'path': str(mock_pdf_dir / "doc1.pdf"),
            'filename': 'doc1.pdf',
            'category': 'pdfs',
            'text': '第一章\nContent one.\n\n第二章\nContent two.\n\n第三章\nContent three.\n\n第四章\nContent four.',
            'char_count': 200
        }

        with patch('skills.unsloth_dataset_creator.get_all_pdf_files', return_value=[
            mock_pdf_dir / "doc1.pdf"
        ]):
            with patch('skills.unsloth_dataset_creator.extract_text_from_pdf', return_value=mock_doc_info):
                with patch('skills.unsloth_dataset_creator.call_llm', return_value=mock_llm_response):
                    result = create_dataset(
                        pdf_dir=str(mock_pdf_dir),
                        output_dir=str(output_dir),
                        config={"use_rag": False, "use_thinking": True, "use_tools": False, "target_samples": 10},
                        llm_provider="groq",
                        llm_config={"api_key": "test_key"}
                    )

        # Check split ratio is approximately 90/10
        train_count = result['metadata']['train_count']
        eval_count = result['metadata']['eval_count']
        total = train_count + eval_count

        if total > 0:
            train_ratio = train_count / total
            assert 0.8 <= train_ratio <= 0.95  # Allow some variance


class TestModuleExports:
    """Test that all expected functions are exported."""

    def test_all_functions_exported(self):
        """Test that all expected functions are available from module."""
        from skills import unsloth_dataset_creator

        assert hasattr(unsloth_dataset_creator, 'create_dataset')
        assert hasattr(unsloth_dataset_creator, 'call_llm')
        assert hasattr(unsloth_dataset_creator, 'chunk_text')
        assert hasattr(unsloth_dataset_creator, 'extract_text_from_pdf')
        assert hasattr(unsloth_dataset_creator, 'get_all_pdf_files')
        assert hasattr(unsloth_dataset_creator, 'create_qa_generation_prompt')
        assert hasattr(unsloth_dataset_creator, 'parse_qa_response')
        assert hasattr(unsloth_dataset_creator, 'validate_qa')
