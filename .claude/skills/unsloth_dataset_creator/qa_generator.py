"""LLM-based Q&A generation for dataset creation."""
import json
from typing import Dict, List, Optional

QUESTION_TYPES = [
    "事実確認型", "手続き型", "条件判断型", "比較型", "具体例型"
]

TYPE_TEMPLATES = {
    "事実確認型": "事実確認型の質問を作成してください。",
    "手続き型": "手続き型の質問を作成してください。",
    "条件判断型": "条件判断型の質問を作成してください。",
    "比較型": "比較型の質問を作成してください。",
    "具体例型": "具体例型の質問を作成してください。"
}

def create_qa_generation_prompt(chunk: Dict, question_type: str, qa_index: int) -> str:
    """Create prompt for Q&A generation."""
    template = TYPE_TEMPLATES.get(question_type, TYPE_TEMPLATES["事実確認型"])
    prompt = f"""以下の規程テキストから、社員が実際に質問しそうな内容とその回答を生成してください。

【規程情報】
ファイル名: {chunk['filename']}
カテゴリ: {chunk['category']}

【規程テキスト】
{chunk['text'][:2000]}{'...' if len(chunk['text']) > 2000 else ''}

【指示】
{template}

【出力形式（JSON）】
{{
  "question": "質問文",
  "answer": "回答文（100文字以上）",
  "thinking": "思考プロセス"
}}

JSON形式で出力してください。"""
    return prompt

def parse_qa_response(response: str) -> Optional[Dict]:
    """Parse LLM response to extract Q&A."""
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()
        qa_dict = json.loads(json_str)
        required = ['question', 'answer', 'thinking']
        if not all(field in qa_dict for field in required):
            return None
        return qa_dict
    except Exception:
        return None

def validate_qa(qa: Dict, min_answer_length: int = 50) -> bool:
    """Validate generated Q&A quality."""
    if len(qa.get('answer', '')) < min_answer_length - 5:
        return False
    if not qa.get('question', '').strip():
        return False
    if not qa.get('answer', '').strip():
        return False
    if not qa.get('thinking', '').strip():
        return False
    return True
