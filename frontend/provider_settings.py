"""Persist non-secret per-task provider choices for the Streamlit UI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict


SETTINGS_FILE = Path(__file__).resolve().parent / "provider_settings.json"
TASKS = ("task1", "task2", "task3")
GROQ_MODELS = {
    "task1": "qwen/qwen3.8-27b",
    "task2": "openai/gpt-oss-20b",
    "task3": "openai/gpt-oss-120b",
}


def _default_provider(task: str) -> str:
    return os.getenv(f"{task.upper()}_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "claude")).strip().lower()


def _default_model(task: str, provider: str) -> str:
    if provider == "groq":
        return os.getenv(f"{task.upper()}_GROQ_MODEL", GROQ_MODELS[task]).strip()
    return os.getenv(f"{task.upper()}_ANTHROPIC_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")).strip()


def model_for(task: str, provider: str) -> str:
    return _default_model(task, provider)


def load_settings() -> Dict[str, Dict[str, str]]:
    result = {task: {"provider": _default_provider(task), "model": _default_model(task, _default_provider(task))} for task in TASKS}
    if not SETTINGS_FILE.exists():
        return result
    try:
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        for task in TASKS:
            if saved.get(task, {}).get("provider") in {"claude", "groq"}:
                result[task]["provider"] = saved[task]["provider"]
                result[task]["model"] = _default_model(task, result[task]["provider"])
    except (OSError, ValueError, TypeError):
        pass
    return result


def save_settings(settings: Dict[str, Dict[str, str]]) -> None:
    safe = {task: {"provider": settings[task]["provider"]} for task in TASKS}
    temporary = SETTINGS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_FILE)
