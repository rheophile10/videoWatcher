# analysts/engine.py
import os
import time
import threading
from typing import Callable, Any, Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
from watcher.llm.resources import register_loaded_model, track_performance

import ollama
from openai import OpenAI
from anthropic import Anthropic

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Global model cache & lock
# ──────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, Any] = {}  # "ollama:llama3.1" → client or True
_CACHE_LOCK = threading.Lock()


def _tqdm_sleep(total_sec: float = 1.5, desc: str = "Warming up model"):
    for _ in tqdm(
        range(100), desc=desc, leave=False, bar_format="{l_bar}{bar}| {n_fmt}%"
    ):
        time.sleep(total_sec / 100)


def clear_model_cache() -> None:
    """Unload all cached models (especially Ollama)"""
    with _CACHE_LOCK:
        for key in list(_MODEL_CACHE.keys()):
            if key.startswith("ollama:"):
                model_name = key.split(":", 1)[1]
                try:
                    ollama.delete(model_name)
                    print(f"Unloaded Ollama model: {model_name}")
                except:
                    pass
        _MODEL_CACHE.clear()
    print("All models unloaded.")


# ──────────────────────────────────────────────────────────────
# Individual completion functions (pure, reusable)
# ──────────────────────────────────────────────────────────────
def _ollama_completion(
    model_name: str,
    messages: List[dict],
    temperature: float,
    stream: bool,
    pretty: bool,
) -> str:
    cache_key = f"ollama:{model_name}"
    with _CACHE_LOCK:
        if cache_key not in _MODEL_CACHE:
            if pretty:
                print(f"Loading Ollama model: {model_name}")
                _tqdm_sleep(2.0, f"Pulling {model_name}")
            try:
                ollama.show(model_name)  # triggers pull
            except:
                pass
            _MODEL_CACHE[cache_key] = True
        elif pretty:
            print(f"Using cached Ollama model: {model_name}")

    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": temperature},
        stream=stream,
    )

    if not stream:
        return response["message"]["content"]

    collected = ""
    if pretty:
        print("Response: ", end="", flush=True)
    for chunk in response:
        content = chunk["message"]["content"]
        collected += content
        if pretty:
            print(content, end="", flush=True)
    if pretty:
        print()
    return collected


def _openai_completion(
    model_name: str,
    messages: List[dict],
    temperature: float,
    stream: bool,
    pretty: bool,
) -> str:
    cache_key = f"openai:{model_name}"
    with _CACHE_LOCK:
        client = _MODEL_CACHE.get(cache_key)
        if not client:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            _MODEL_CACHE[cache_key] = client

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        stream=stream,
    )

    if not stream:
        return resp.choices[0].message.content

    collected = ""
    if pretty:
        print("Response: ", end="", flush=True)
    for chunk in resp:
        if chunk.choices and chunk.choices[0].delta.content:
            c = chunk.choices[0].delta.content
            collected += c
            if pretty:
                print(c, end="", flush=True)
    if pretty:
        print()
    return collected


def _claude_completion(
    model_name: str,
    messages: List[dict],
    temperature: float,
    stream: bool,
    pretty: bool,
) -> str:
    cache_key = f"claude:{model_name}"
    with _CACHE_LOCK:
        client = _MODEL_CACHE.get(cache_key)
        if not client:
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            _MODEL_CACHE[cache_key] = client

    resp = client.messages.create(
        model=model_name,
        max_tokens=4096,
        temperature=temperature,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        stream=stream,
    )

    if not stream:
        return resp.content[0].text

    collected = ""
    if pretty:
        print("Response: ", end="", flush=True)
    for chunk in resp:
        if chunk.type == "content_block_delta" and getattr(chunk.delta, "text", None):
            c = chunk.delta.text
            collected += c
            if pretty:
                print(c, end="", flush=True)
    if pretty:
        print()
    return collected


def _grok_completion(
    model_name: str,
    messages: List[dict],
    temperature: float,
    stream: bool,
    pretty: bool,
) -> str:
    # Grok uses OpenAI-compatible endpoint
    cache_key = "grok:grok-beta"
    with _CACHE_LOCK:
        client = _MODEL_CACHE.get(cache_key)
        if not client:
            client = OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
            _MODEL_CACHE[cache_key] = client

    resp = client.chat.completions.create(
        model="grok-beta",
        messages=messages,
        temperature=temperature,
        stream=False,  # Grok doesn't support streaming yet
    )
    return resp.choices[0].message.content


# ──────────────────────────────────────────────────────────────
# Dispatch dictionary — the real "switch statement"
# ──────────────────────────────────────────────────────────────
COMPLETION_FUNCTIONS: Dict[str, Callable] = {
    "ollama": _ollama_completion,
    "openai": _openai_completion,
    "claude": _claude_completion,
    "grok": _grok_completion,
}


# ──────────────────────────────────────────────────────────────
# Public unified API
# ──────────────────────────────────────────────────────────────
def chat_completion(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.1,
    stream: bool = True,
    pretty: bool = True,
    return_perf: bool = False,
) -> tuple[str, dict] | str:
    engine_name = model.split(":", 1)[0]
    model_name = model[len(engine_name) + 1 :]

    if engine_name == "ollama":
        register_loaded_model(model)

    def _run():
        return COMPLETION_FUNCTIONS[engine_name](
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream,
            pretty=pretty,
        )

    if return_perf:
        import time

        start_time = time.time()
        response = _run()
        end_time = time.time()
        perf = {
            "time": end_time - start_time,
            "model": model,
            "engine": engine_name,
        }
        return response, perf

    return _run()
