"""Drafts grounded replies using Task 1 evidence and selected LLM."""
import logging
from typing import Any, Dict, Optional

from .classifier import classify_email
from .knowledge_client import KnowledgeClient, KnowledgeResult
from .llm_provider import BaseLLMProvider
from task2_inbox_agent.learning.memory_store import MemoryStore

logger = logging.getLogger(__name__)

SYSTEM = """You draft professional customer email replies. Use only the supplied Task 1 evidence.
Never invent product features, prices, refunds, timelines, or policy. If evidence is missing,
say that a specialist must review the request. Return only the reply text."""


class DraftingAgent:
    def __init__(self, knowledge: Optional[KnowledgeClient] = None,
                 llm: Optional[BaseLLMProvider] = None,
                 memory: Optional[MemoryStore] = None):
        self.knowledge = knowledge or KnowledgeClient()
        self.llm = llm
        self.memory = memory or MemoryStore()

    def draft(self, email_id: str, sender: str, subject: str, body: str) -> Dict[str, Any]:
        category = classify_email(subject, body, self.llm)
        evidence: KnowledgeResult = self.knowledge.query(f"{subject}\n{body}", top_k=4)
        feedback = self.memory.get_feedback_for_category(category, limit=3)
        prompt = self._prompt(sender, subject, body, category, evidence, feedback)
        provider_error = None
        model_used = getattr(self.llm, "model_name", "none") if self.llm else "deterministic"
        provider = getattr(self.llm, "provider_label", "deterministic") if self.llm else "deterministic"
        if evidence.has_answer and evidence.answer.strip():
            if self.llm:
                try:
                    result = self.llm.call(prompt, system_prompt=SYSTEM)
                    text = str(result.get("text", "")).strip()
                    if not text:
                        raise RuntimeError("provider returned empty text")
                    input_tokens, output_tokens = result.get("input_tokens"), result.get("output_tokens")
                except Exception as exc:
                    provider_error = f"{type(exc).__name__}: {exc}"
                    text = self._safe_reply(evidence)
                    provider = "fallback"
                    model_used = "deterministic_fallback"
                    input_tokens = output_tokens = None
            else:
                text = self._safe_reply(evidence)
                input_tokens = output_tokens = None
        else:
            text = ("Thank you for contacting us. We do not have enough verified product "
                    "information to answer this request. A specialist will review it and "
                    "follow up with you.")
            provider = "not_used"
            model_used = "not_used"
            input_tokens = output_tokens = None
        return {"email_id": email_id, "category": category, "draft": text,
                "knowledge": evidence, "provider": provider, "model_used": model_used,
                "provider_error": provider_error, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "requires_human": (
                    not evidence.has_answer or category in {
                        "refund_request", "complaint", "legal", "escalation",
                        "new_project_request", "quote_request", "follow_up"})}

    def _prompt(self, sender, subject, body, category, evidence, feedback):
        examples = "\n".join(f"Previous correction: {x.final_reply}" for x in feedback)
        return (f"Customer: {sender}\nSubject: {subject}\nBody: {body}\nCategory: {category}\n"
                f"Task 1 answer: {evidence.answer}\nCitations: {evidence.citations}\n"
                f"Relevant feedback examples:\n{examples}")

    @staticmethod
    def _safe_reply(evidence):
        return evidence.answer.strip() + "\n\nPlease reply if you need further help."
