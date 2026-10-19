# -*- coding: utf-8 -*-
"""
LLM Provider abstraction.
Provides a common interface for Claude (Anthropic) and Groq models.
The provider is selected via the LLM_PROVIDER environment variable (default: 'claude').
All calls return a dict containing the generated text and optional token usage.
"""

import os
import logging
from typing import Dict, Any
import httpx

logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`call` and expose ``model_name`` attribute.
    The ``call`` method should return a dict with keys ``text``, ``input_tokens``
    and ``output_tokens`` when the provider supplies them.
    """

    model_name: str = "unknown"

    def __init__(self) -> None:
        pass

    def call(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ClaudeProvider(BaseLLMProvider):
    """Claude (Anthropic) implementation.

    Uses the ``ANTHROPIC_API_KEY`` and the ``ANTHROPIC_MODEL`` environment
    variables. Errors are propagated as ``httpx.HTTPError`` subclasses which
    the caller can interpret.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model_name = (model_name or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")).strip()
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set – Claude calls will fail")

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "max_tokens": 1024,
            "system": """You are a helpful assistant that must answer strictly based on the provided documentation. Do not fabricate information. If the documentation does not contain the answer, respond with \"I don't know based on the provided documentation.\"""",
            "messages": [{"role": "user", "content": prompt}],
        }

    def call(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = self._build_payload(prompt)
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # Extract text content
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
    """Groq implementation.

    Uses ``GROQ_API_KEY`` and ``GROQ_MODEL`` environment variables.
    The Groq API follows the OpenAI compatible schema.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = (model_name or os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")).strip()
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set – Groq calls will fail")

    def call(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1024,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI style response parsing
            choice = data.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }


def get_llm_provider(provider_name: str | None = None, model_name: str | None = None) -> BaseLLMProvider:
    """Factory that returns the configured LLM provider.

    The ``LLM_PROVIDER`` environment variable determines which concrete class
    is instantiated. Supported values are ``"claude"`` and ``"groq"``.
    The function never silently falls back – an unsupported value raises a
    ``ValueError``.
    """
    selected = (provider_name or os.getenv("LLM_PROVIDER", "claude")).strip().lower()
    if selected == "claude":
        return ClaudeProvider(model_name)
    if selected == "groq":
        return GroqProvider(model_name)
    raise ValueError(f"Unsupported LLM provider: {selected}")
