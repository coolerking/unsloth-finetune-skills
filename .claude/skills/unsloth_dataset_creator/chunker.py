"""Text chunking utilities for dataset creation."""
from typing import Dict, List
import tiktoken

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_tokenizer.encode(text))


def split_by_sections(text: str, section_keywords: List[str] = None) -> List[str]:
    """Split text by section markers."""
    if section_keywords is None:
        section_keywords = ["第", "条", "章", "節", "項"]
    sections = []
    current_section = []
    lines = text.split('\n')
    for line in lines:
        is_section_start = any(keyword in line for keyword in section_keywords)
        if is_section_start and current_section:
            sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    if current_section:
        sections.append('\n'.join(current_section))
    return sections


def split_by_tokens(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += (chunk_size - overlap)
    return chunks


def chunk_text(
    doc_info: Dict,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    section_keywords: List[str] = None
) -> List[Dict]:
    """Hybrid chunking: section-based then token-based."""
    text = doc_info['text']
    chunks = []
    sections = split_by_sections(text, section_keywords)
    for idx, section in enumerate(sections):
        token_count = count_tokens(section)
        if token_count > chunk_size:
            sub_chunks = split_by_tokens(section, chunk_size, chunk_overlap)
            for sub_idx, sub_chunk in enumerate(sub_chunks):
                chunks.append({
                    'text': sub_chunk,
                    'filename': doc_info['filename'],
                    'category': doc_info['category'],
                    'chunk_id': f"{doc_info['filename']}_sec{idx}_sub{sub_idx}",
                    'token_count': count_tokens(sub_chunk)
                })
        else:
            chunks.append({
                'text': section,
                'filename': doc_info['filename'],
                'category': doc_info['category'],
                'chunk_id': f"{doc_info['filename']}_sec{idx}",
                'token_count': token_count
            })
    return chunks
