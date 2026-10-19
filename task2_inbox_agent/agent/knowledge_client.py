"""
HTTP client that calls the Task 1 Knowledge Brain API to retrieve
product/platform documentation for answering customer emails.
Does NOT copy or embed the RAG / ChromaDB logic — strictly uses the REST API.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from task2_inbox_agent.config import TASK1_QUERY_ENDPOINT

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    """Encapsulates the response from Task 1 /api/v1/query."""
    query: str
    answer: str
    has_answer: bool
    confidence: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    ticket_id: Optional[str] = None
    model_used: str = "none"
    fallback_used: bool = False
    generation_error: Optional[str] = None


class KnowledgeClient:
    """
    Calls the Task 1 REST API.
    If Task 1 is unreachable, returns a ``KnowledgeResult`` with
    ``has_answer=False`` so the drafting agent can handle it gracefully.
    """

    def __init__(self, base_url: Optional[str] = None, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        self.endpoint = base_url or TASK1_QUERY_ENDPOINT
        self.provider = provider
        self.model = model

    def query(self, question: str, top_k: int = 4, category: Optional[str] = None) -> KnowledgeResult:
        payload: Dict[str, Any] = {"query": question, "top_k": top_k}
        if category:
            payload["category_filter"] = category
        if self.provider:
            payload["provider"] = self.provider
        if self.model:
            payload["model"] = self.model
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return KnowledgeResult(
                    query=data.get("query", question),
                    answer=data.get("answer", ""),
                    has_answer=data.get("has_answer", False),
                    confidence=data.get("confidence", 0.0),
                    citations=data.get("citations", []),
                    screenshots=data.get("screenshots", []),
                    ticket_id=data.get("ticket_id"),
                    model_used=data.get("model_used", "none"),
                    fallback_used=data.get("fallback_used", False),
                    generation_error=data.get("generation_error"),
                )
        except Exception as exc:
            logger.error("Task 1 API call failed: %s", exc)
            return KnowledgeResult(
                query=question,
                answer="",
                has_answer=False,
                confidence=0.0,
                generation_error=f"Task 1 API unreachable: {type(exc).__name__}: {exc}",
            )
