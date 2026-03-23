from skills.unsloth_dataset_creator.qa_generator import (
    create_qa_generation_prompt,
    parse_qa_response,
    validate_qa
)

def test_create_qa_generation_prompt():
    chunk = {
        'text': 'Test content',
        'filename': 'test.pdf',
        'category': 'hr',
        'chunk_id': 'test_sec0'
    }
    prompt = create_qa_generation_prompt(chunk, "事実確認型", 0)
    assert "Test content" in prompt
    assert "事実確認型" in prompt
    assert "JSON形式" in prompt

def test_parse_qa_response_valid():
    response = '{"question": "What?", "answer": "This is a test.", "thinking": "I analyzed."}'
    result = parse_qa_response(response)
    assert result['question'] == "What?"
    assert result['answer'] == "This is a test."

def test_parse_qa_response_with_code_block():
    response = '''```json\n{"question": "Q?", "answer": "A.", "thinking": "T."}\n```'''
    result = parse_qa_response(response)
    assert result['question'] == "Q?"

def test_parse_qa_response_invalid():
    response = "Not valid JSON"
    result = parse_qa_response(response)
    assert result is None

def test_validate_qa_valid():
    qa = {
        'question': 'Test?',
        'answer': 'Detailed answer with more than fifty characters.',
        'thinking': 'Analyzed regulation section 1.'
    }
    assert validate_qa(qa, min_answer_length=50) is True

def test_validate_qa_too_short():
    qa = {'question': 'Test?', 'answer': 'Short.', 'thinking': 'Analysis.'}
    assert validate_qa(qa, min_answer_length=50) is False
