"""
Email classifier — categorises incoming customer emails.
Uses an LLM for nuanced classification but includes a fast keyword-based
fallback so the system works even without API credits.
"""

import logging
import re
from typing import Optional

from task2_inbox_agent.agent.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

# Canonical categories the system supports
CATEGORIES = [
    "product_support",
    "new_project_request",
    "refund_request",
    "complaint",
    "quote_request",
    "follow_up",
    "legal",
    "general_inquiry",
]

SENSITIVE_CATEGORIES = {"refund_request", "complaint", "legal", "escalation"}

_CLASSIFICATION_SYSTEM = """You are an email classification assistant for an AI tech consultancy that builds websites using Brilliant Directories.
Classify the customer email into exactly ONE of these categories:
- product_support: questions about product features, settings, how-to
- new_project_request: wants to start a new website or project
- refund_request: asks for a refund or cancellation
- complaint: expresses dissatisfaction, anger, or frustration
- quote_request: requests a price quote or estimate
- follow_up: references a previous conversation, ticket, or project
- legal: legal dispute, legal advice, or legal terms
- general_inquiry: anything else (greetings, partnerships, etc.)

Respond with ONLY the category slug (e.g. "product_support"). No explanation."""

# Simple keyword-based fallback
_KEYWORD_MAP = {
    "refund": "refund_request",
    "money back": "refund_request",
    "cancel": "refund_request",
    "cancellation": "refund_request",
    "quote": "quote_request",
    "pricing": "quote_request",
    "estimate": "quote_request",
    "cost": "quote_request",
    "how much": "quote_request",
    "budget": "quote_request",
    "new project": "new_project_request",
    "new website": "new_project_request",
    "build a website": "new_project_request",
    "start a project": "new_project_request",
    "want to build": "new_project_request",
    "following up": "follow_up",
    "follow up": "follow_up",
    "previous email": "follow_up",
    "ticket": "follow_up",
    "last week": "follow_up",
    "update on": "follow_up",
    "legal": "legal",
    "lawsuit": "legal",
    "dispute": "legal",
    "attorney": "legal",
    "disappointed": "complaint",
    "frustrated": "complaint",
    "unacceptable": "complaint",
    "terrible": "complaint",
    "worst": "complaint",
    "angry": "complaint",
    "unhappy": "complaint",
    "how to": "product_support",
    "how do i": "product_support",
    "setting": "product_support",
    "configure": "product_support",
    "feature": "product_support",
    "not working": "product_support",
    "error": "product_support",
    "bug": "product_support",
    "issue": "product_support",
}


def _keyword_classify(text: str) -> str:
    """Fast keyword-based fallback."""
    lower = text.lower()
    for keyword, cat in _KEYWORD_MAP.items():
        if keyword in lower:
            return cat
    return "general_inquiry"


def classify_email(
    subject: str,
    body: str,
    llm: Optional[BaseLLMProvider] = None,
) -> str:
    """
    Returns a canonical category string.
    Uses the LLM when available; falls back to keyword matching otherwise.
    """
    combined = f"{subject}\n{body}"
    if llm is not None:
        try:
            prompt = f"Subject: {subject}\n\nBody:\n{body}"
            result = llm.call(prompt, system_prompt=_CLASSIFICATION_SYSTEM)
            raw = result["text"].strip().lower().replace('"', "").replace("'", "")
            # The LLM might return extra words; grab first valid slug
            for cat in CATEGORIES:
                if cat in raw:
                    return cat
            logger.warning("LLM returned unrecognised category %r; falling back to keywords", raw)
        except Exception as exc:
            logger.warning("LLM classification failed (%s); using keyword fallback", exc)
    return _keyword_classify(combined)
