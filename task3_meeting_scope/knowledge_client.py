import httpx
from typing import Any, Dict
from .config import TASK1_API_BASE

class KnowledgeClient:
    def __init__(self, base_url: str = TASK1_API_BASE, provider: str | None = None, model: str | None = None): self.endpoint=base_url.rstrip("/")+"/api/v1/query"; self.provider=provider; self.model=model
    def query(self, question: str) -> Dict[str, Any]:
        try:
            payload={"query":question,"top_k":4}
            if self.provider: payload["provider"] = self.provider
            if self.model: payload["model"] = self.model
            r=httpx.post(self.endpoint,json=payload,timeout=30); r.raise_for_status(); return r.json()
        except Exception as exc:
            return {"has_answer":False,"answer":"","citations":[],"generation_error":f"Task 1 unavailable: {type(exc).__name__}: {exc}"}
