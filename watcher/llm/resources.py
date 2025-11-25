"""track system resources and recommend models accordingly

recommended use example

from analysts import chat_completion, recommend_model, unload_model

# Smart mode
model = recommend_model(max_vram_gb=20)
print(f"Using smart model: {model}")

response, perf = chat_completion(
    model=model,
    messages=[{"role": "user", "content": "Analyze this..."}],
    return_perf=True
)

print("Performance:", perf)

# Later: free VRAM
unload_model("ollama:llama3.1:70b-instruct-q4_K_M")
"""

import psutil
import torch
import time
from typing import Dict, Callable
from contextlib import contextmanager
import ollama

# Global tracking
_MODEL_MEMORY_FOOTPRINTS = {
    # Known model → estimated VRAM (GB) when loaded
    "whisper:large-v2": 2.4,
    "whisper:large-v3": 2.9,
    "pyannote/speaker-diarization-3.1": 1.1,
    "bge-large-en-v1.5": 1.3,
    "ollama:llama3.1:8b": 6.0,
    "ollama:llama3.1:70b": 38.0,
    "ollama:gemma2:27b": 18.0,
    "ollama:mixtral:8x22b": 48.0,
}

# Track what we manually loaded
_LOADED_RESOURCES: Dict[str, float] = {}  # name → estimated GB


def get_system_memory() -> Dict[str, float]:
    """Returns RAM and VRAM in GB"""
    ram = psutil.virtual_memory()
    vram = {"total": 0.0, "used": 0.0, "free": 0.0}
    if torch.cuda.is_available():
        vram["total"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram["used"] = torch.cuda.memory_allocated(0) / 1e9
        vram["free"] = vram["total"] - vram["used"]
    return {
        "ram_total_gb": ram.total / 1e9,
        "ram_used_gb": ram.used / 1e9,
        "ram_free_gb": ram.available / 1e9,
        "vram_total_gb": vram["total"],
        "vram_used_gb": vram["used"],
        "vram_free_gb": vram["free"],
    }


def estimate_model_vram(model_key: str) -> float:
    """Best guess VRAM usage in GB"""
    return _MODEL_MEMORY_FOOTPRINTS.get(model_key, 10.0)  # default conservative


def register_loaded_model(name: str, estimated_gb: float = None):
    if estimated_gb is None:
        estimated_gb = estimate_model_vram(name)
    _LOADED_RESOURCES[name] = estimated_gb


def unregister_model(name: str):
    _LOADED_RESOURCES.pop(name, None)


def get_loaded_models() -> Dict[str, float]:
    return _LOADED_RESOURCES.copy()


def track_performance(func):
    """Decorator or simple wrapper to measure execution + memory"""

    def wrapper(*args, **kwargs):
        before = get_system_memory()
        peak_vram = before["vram_used_gb"]
        peak_ram = before["ram_used_gb"]
        start = time.time()

        result = func(*args, **kwargs)

        duration = time.time() - start
        after = get_system_memory()

        # Sample a few times post-execution (catches delayed GC)
        for _ in range(8):
            current = get_system_memory()["vram_used_gb"]
            peak_vram = max(peak_vram, current)
            time.sleep(0.02)

        perf = {
            "duration_sec": round(duration, 3),
            "ram_before_gb": round(before["ram_used_gb"], 2),
            "ram_after_gb": round(after["ram_used_gb"], 2),
            "ram_peak_gb": round(max(peak_ram, after["ram_used_gb"]), 2),
            "vram_before_gb": round(before["vram_used_gb"], 2),
            "vram_after_gb": round(after["vram_used_gb"], 2),
            "vram_peak_gb": round(peak_vram, 2),
            "loaded_models": get_loaded_models(),
        }
        return result, perf

    return wrapper


def recommend_model(
    max_vram_gb: float = 15.9,
    max_ram_gb: float = 31.8,
    prefer_local: bool = True,
) -> str:
    """Suggest best model given current resource limits"""
    current = get_system_memory()
    loaded_vram = sum(get_loaded_models().values())

    available_vram = current["vram_free_gb"] + loaded_vram  # assume we can unload
    available_ram = current["ram_free_gb"]

    candidates = [
        ("ollama:llama3.1:70b-instruct-q4_K_M", 38.0),
        ("ollama:gemma2:27b-instruct-q4_K_M", 18.0),
        ("ollama:llama3.1:8b-instruct-q4_K_M", 6.0),
        ("openai:gpt-4o", 0.0),
        ("claude:claude-3-5-sonnet-20241022", 0.0),
    ]

    for model_str, vram_cost in candidates:
        if vram_cost > 0 and vram_cost > available_vram:
            continue
        if vram_cost == 0 and not prefer_local:
            continue
        if model_str.startswith("openai") or model_str.startswith("claude"):
            return model_str
        if vram_cost <= max_vram_gb and available_ram > 10:
            return model_str

    return "openai:gpt-4o-mini"  # safe fallback


def unload_model(name: str):
    """Unload any tracked model"""
    if name.startswith("ollama:"):
        try:
            ollama.delete(name.split(":", 1)[1])
            print(f"Unloaded {name}")
        except:
            pass
    elif name.startswith("whisper:"):
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    elif name == "pyannote/speaker-diarization-3.1":
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    unregister_model(name)
