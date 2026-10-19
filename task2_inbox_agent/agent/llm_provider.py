"""
LLM Provider for Task 2 Inbox Agent.
Reuses the same Claude/Groq abstraction as Task 1 but supports a system prompt,
which is essential for email drafting instructions.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Common interface for LLM backends."""

    model_name: str = "unknown"
    provider_label: str = "unknown"

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Return {"text": ..., "input_tokens": ..., "output_tokens": ...}."""
        raise NotImplementedError


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude via REST."""

    provider_label = "claude"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model_name = (model_name or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")).strip()
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — Claude calls will fail")

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = "\n".join(
                block["text"]
                for block in data.get("content", [])
                if block.get("type") == "text"
            )
            usage = data.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }


class GroqProvider(BaseLLMProvider):
    """Groq (OpenAI-compatible) via REST."""

    provider_label = "groq"

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = (model_name or os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")).strip()
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set — Groq calls will fail")

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
        }
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }


def get_llm_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory — returns the requested LLM provider.
    Falls back to the LLM_PROVIDER env var (default ``groq``).
    Never silently switches on failure.
    """
    name = (provider_name or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
    if name == "claude":
        return ClaudeProvider(model_name)
    if name == "groq":
        return GroqProvider(model_name)
    raise ValueError(f"Unsupported LLM provider: {name!r}")
