"""Unsloth Dataset Creator - Generate training datasets from PDF documents.

This module provides functionality to:
- Extract text from PDF files
- Chunk documents using a hybrid approach (section-based + token-based)
- Generate Q&A pairs using LLM
- Create train/eval splits
- Output datasets in JSONL format
"""
import json
import os
import random
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import from sibling modules
from .pdf_processor import extract_text_from_pdf, get_all_pdf_files
from .chunker import chunk_text
from .qa_generator import create_qa_generation_prompt, parse_qa_response, validate_qa, QUESTION_TYPES

__all__ = [
    'create_dataset',
    'call_llm',
    'chunk_text',
    'extract_text_from_pdf',
    'get_all_pdf_files',
    'create_qa_generation_prompt',
    'parse_qa_response',
    'validate_qa',
]


def call_llm(provider: str, prompt: str, llm_config: Dict[str, Any]) -> Optional[Dict]:
    """Call LLM provider to generate Q&A.

    Args:
        provider: LLM provider name (e.g., "groq")
        prompt: The prompt to send to the LLM
        llm_config: Configuration for the LLM including api_key, model, etc.

    Returns:
        Parsed Q&A dictionary or None if failed
    """
    if provider != "groq":
        raise ValueError(f"Unsupported provider: {provider}. Only 'groq' is supported.")

    api_key = llm_config.get("api_key")
    if not api_key:
        raise ValueError("api_key is required in llm_config")

    model = llm_config.get("model", "llama-3.1-8b-instant")
    temperature = llm_config.get("temperature", 0.7)
    max_tokens = llm_config.get("max_tokens", 2048)

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates Q&A pairs from document content. Always respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            return parse_qa_response(content)
        return None
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def create_dataset(
    pdf_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    llm_provider: str,
    llm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a training dataset from PDF documents.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to write output files
        config: Configuration dict with use_thinking, target_samples
        llm_provider: LLM provider name (e.g., "groq")
        llm_config: LLM configuration dict

    Returns:
        Status dictionary with 'status', 'message', and 'metadata' keys
    """
    pdf_path = Path(pdf_dir)
    output_path = Path(output_dir)

    # Validate inputs
    if not pdf_path.exists():
        return {
            'status': 'error',
            'message': f'PDF directory does not exist: {pdf_dir}',
            'metadata': {}
        }

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PDF files
    pdf_files = get_all_pdf_files(pdf_path)
    if not pdf_files:
        return {
            'status': 'error',
            'message': f'No PDF files found in: {pdf_dir}',
            'metadata': {}
        }

    # Extract text from all PDFs
    documents = []
    for pdf_file in pdf_files:
        doc_info = extract_text_from_pdf(pdf_file)
        if doc_info:
            documents.append(doc_info)

    if not documents:
        return {
            'status': 'error',
            'message': 'Failed to extract text from any PDF files',
            'metadata': {}
        }

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            'status': 'error',
            'message': 'No chunks generated from documents',
            'metadata': {}
        }

    # Generate Q&A pairs
    use_thinking = config.get('use_thinking', True)
    target_samples = config.get('target_samples', 100)

    qa_pairs = []
    max_attempts = min(len(all_chunks) * len(QUESTION_TYPES), target_samples * 3)
    attempts = 0

    # Shuffle chunks for variety
    random.shuffle(all_chunks)

    for chunk in all_chunks:
        if len(qa_pairs) >= target_samples or attempts >= max_attempts:
            break

        for q_type in QUESTION_TYPES:
            if len(qa_pairs) >= target_samples or attempts >= max_attempts:
                break

            prompt = create_qa_generation_prompt(chunk, q_type, len(qa_pairs))
            qa_response = call_llm(llm_provider, prompt, llm_config)

            attempts += 1

            if qa_response and validate_qa(qa_response):
                qa_entry = {
                    'instruction': qa_response['question'],
                    'output': qa_response['answer'],
                    'thinking': qa_response.get('thinking', '') if use_thinking else '',
                    'metadata': {
                        'source_file': chunk['filename'],
                        'category': chunk['category'],
                        'chunk_id': chunk['chunk_id'],
                        'question_type': q_type,
                        'token_count': chunk['token_count']
                    }
                }
                qa_pairs.append(qa_entry)

    if not qa_pairs:
        return {
            'status': 'error',
            'message': 'Failed to generate any valid Q&A pairs',
            'metadata': {}
        }

    # Split train/eval (90/10)
    random.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * 0.9)
    train_pairs = qa_pairs[:split_idx]
    eval_pairs = qa_pairs[split_idx:]

    # Write output files
    train_path = output_path / "dataset_train.jsonl"
    eval_path = output_path / "dataset_eval.jsonl"
    metadata_path = output_path / "dataset_metadata.json"

    # Write train file
    with open(train_path, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # Write eval file
    with open(eval_path, 'w', encoding='utf-8') as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    # Write metadata
    metadata = {
        'total_samples': len(qa_pairs),
        'train_count': len(train_pairs),
        'eval_count': len(eval_pairs),
        'source_documents': len(documents),
        'total_chunks': len(all_chunks),
        'config': config,
        'llm_provider': llm_provider,
        'llm_model': llm_config.get('model', 'unknown')
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        'status': 'success',
        'message': f'Successfully created dataset with {len(qa_pairs)} samples',
        'metadata': metadata
    }
