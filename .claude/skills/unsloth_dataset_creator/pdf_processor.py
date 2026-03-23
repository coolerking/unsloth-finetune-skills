"""PDF text extraction utilities."""
from pathlib import Path
from typing import Dict, List, Optional
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path: Path) -> Optional[Dict[str, any]]:
    """Extract text from a PDF file."""
    try:
        text = extract_text(str(pdf_path))
        category = pdf_path.parent.name
        return {
            'path': str(pdf_path),
            'filename': pdf_path.name,
            'category': category,
            'text': text.strip(),
            'char_count': len(text.strip())
        }
    except Exception as e:
        print(f"Error extracting {pdf_path.name}: {e}")
        return None

def get_all_pdf_files(root_dir: Path) -> List[Path]:
    """Get all PDF files recursively, excluding checkpoints."""
    pdf_files = list(root_dir.rglob("*.pdf"))
    pdf_files = [f for f in pdf_files if '.ipynb_checkpoints' not in str(f)]
    return sorted(pdf_files)
