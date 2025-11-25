"""LLM prompt execution module.
example use
from analysts import assess_videos, assess_chunks, generate_memos, clear_model_cache

videos = [{"id": 1, "title": "...", "description": "..."}]
results = assess_videos(videos, model="ollama:llama3.1:70b-instruct-q4_K_M")

chunks = [{"id": 5, "speaker": "...", "text": "..."}]
results = assess_chunks(chunks, keywords=["gun", "firearm"], model="openai:gpt-4o")

memos = generate_memos(results, platform="X", model="claude:claude-3-5-sonnet-20241022")

clear_model_cache()  # When done

"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any
from .engine import chat_completion, clear_model_cache

PACKAGE_DIR = Path(__file__).parent
PROMPTS_FILE = PACKAGE_DIR / "prompts.txt"

_prompt_cache: Dict[str, str] = {}


def _load_prompts() -> Dict[str, str]:
    if _prompt_cache:
        return _prompt_cache

    content = PROMPTS_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"--\s*prompt_name:\s*(\w+)", content)[1:]
    for name, text in zip(blocks[::2], blocks[1::2]):
        _prompt_cache[name.strip()] = text.strip()
    return _prompt_cache


def execute_prompt(
    prompt_name: str,
    params: Dict[str, Any],
    *,
    model: str = "ollama:llama3.1:70b-instruct-q4_K_M",
    temperature: float = 0.1,
) -> List[Dict[str, Any]]:
    prompts = _load_prompts()
    template = prompts.get(prompt_name)
    if not template:
        raise KeyError(f"Prompt '{prompt_name}' not found")

    # Render {{var}}
    prompt_text = template
    for key, value in params.items():
        prompt_text = prompt_text.replace(f"{{{{{key}}}}}", str(value))

    messages = [{"role": "user", "content": prompt_text}]

    raw = chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        pretty=True,
    )

    # Clean JSON
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
        return data
    except json.JSONDecodeError as e:
        print("JSON parse failed:", e)
        print("Raw output:", raw[:1000])
        return params.get("records", [])


# High-level API (same as before)
def assess_videos(
    records: List[Dict], model: str = "ollama:llama3.1:70b-instruct-q4_K_M"
) -> List[Dict]:
    return execute_prompt(
        "video_title_assessor",
        {"records": json.dumps(records, ensure_ascii=False)},
        model=model,
    )


def assess_chunks(
    records: List[Dict],
    keywords: List[str],
    model: str = "ollama:llama3.1:70b-instruct-q4_K_M",
) -> List[Dict]:
    return execute_prompt(
        "chunk_relevance_assessor",
        {
            "records": json.dumps(records, ensure_ascii=False),
            "keywords": json.dumps(keywords, ensure_ascii=False),
        },
        model=model,
    )


def generate_memos(
    chunks: List[Dict],
    platform: str = "X",
    model: str = "ollama:llama3.1:70b-instruct-q4_K_M",
) -> List[Dict]:
    return execute_prompt(
        "content_memo_generator",
        {
            "records": json.dumps(chunks, ensure_ascii=False),
            "platform": platform,
        },
        model=model,
    )


__all__ = [
    "assess_videos",
    "assess_chunks",
    "generate_memos",
    "execute_prompt",
    "clear_model_cache",
]
