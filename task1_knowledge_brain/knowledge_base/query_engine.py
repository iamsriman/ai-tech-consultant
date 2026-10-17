"""
RAG Query Engine for Task 1: Knowledge Brain.
Retrieves grounded documentation from ChromaDB, queries Claude with strict
honesty guardrails, provides citations and screenshots, and falls back to
human expert ticketing when evidence is insufficient.
"""

from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import httpx

from .retriever import KnowledgeRetriever, RetrievedDocument
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager, ExpertTicket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the AI Knowledge Brain for Brilliant Directories Solution Center.
Your role is to answer questions accurately and honestly based SOLELY on the provided documentation sources.

STRICT RULES:
1. Grounding: Answer strictly using facts from the provided sources. Do NOT make up settings, menus, or features.
2. Honesty: If the documentation does not contain enough information to answer the question, state honestly:
   "I don't know based on the provided documentation."
3. Citations: At the end of your response, list the exact Article Title and URL where the answer is found.
4. Screenshots: If the sources include screenshots that show the UI steps, explicitly mention and link the screenshot URLs so users can see the exact settings in their admin area.
5. Tone: Concise, professional, and clear for founders and webmasters.
"""


@dataclass
class QueryAnswer:
    query: str
    answer: str
    has_answer: bool
    confidence: float
    citations: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    ticket_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "has_answer": self.has_answer,
            "confidence": self.confidence,
            "citations": self.citations,
            "screenshots": self.screenshots,
            "ticket_id": self.ticket_id,
        }


class BDQueryEngine:
    """
    Executes grounded RAG questions against ChromaDB and Claude.
    """

    def __init__(
        self,
        retriever: Optional[KnowledgeRetriever] = None,
        ticket_manager: Optional[TicketManager] = None,
        vector_store: Optional["BDVectorStore"] = None,
        model_name: str = "claude-3-5-haiku-20241022",
        min_confidence_threshold: float = 0.35,
    ):
        # Backward compatibility: tests may pass vector_store directly
        if vector_store is not None and retriever is None:
            retriever = KnowledgeRetriever(vector_store)
        self.retriever = retriever or KnowledgeRetriever()
        self.ticket_manager = ticket_manager or TicketManager()
        self.model_name = model_name
        self.min_confidence_threshold = min_confidence_threshold
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.retriever = retriever or KnowledgeRetriever()
        self.ticket_manager = ticket_manager or TicketManager()
        self.model_name = model_name
        self.min_confidence_threshold = min_confidence_threshold
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    def _call_claude(self, prompt: str) -> str:
        """Calls Anthropic Claude API via httpx."""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    def _deterministic_grounded_answer(
        self,
        query: str,
        docs: List[RetrievedDocument],
    ) -> str:
        """
        Deterministic, grounded response generator when running offline or without an API key.
        Synthesizes facts directly from the retrieved documentation context.
        Strictly enforces honesty guardrails by filtering stop words.
        """
        top_doc = docs[0]
        stop_words = {
            "how", "do", "i", "a", "an", "the", "in", "on", "to", "for", "of", "and",
            "or", "is", "are", "using", "with", "can", "it", "this", "that", "at",
            "from", "by", "my", "we", "you", "your", "what", "where", "which", "into",
            "brilliant", "directories"
        }
        query_content_words = {
            w for w in re.findall(r"\w+", query.lower()) if len(w) > 2 and w not in stop_words
        }
        doc_words = {
            w for w in re.findall(r"\w+", top_doc.text.lower()) if len(w) > 2
        }
        meaningful_overlap = query_content_words.intersection(doc_words)

        # If key query terms don't appear in the retrieved doc, honestly admit lack of evidence
        if len(meaningful_overlap) < 1 or (len(meaningful_overlap) < 2 and top_doc.score < 0.65):
            return "I don't know based on the provided documentation."

        lines = [f"Based on the Brilliant Directories documentation for **{top_doc.title}**:"]
        
        # Extract main guidance paragraphs
        clean_text = top_doc.text.split("[Associated UI Screenshots]:")[0]
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip() and not p.startswith("[Article:")]
        if paragraphs:
            lines.append("\n" + "\n\n".join(paragraphs[:3]))

        if top_doc.screenshot_urls:
            lines.append("\n**Helpful Screenshot(s):**")
            for idx, s in enumerate(top_doc.screenshot_urls[:2], 1):
                lines.append(f"- [View UI Screenshot {idx}]({s})")

        lines.append(f"\n**Source:** [{top_doc.title}]({top_doc.url}) (Category: {top_doc.category} > {top_doc.folder})")
        return "\n".join(lines)

    def query(
        self,
        question: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
    ) -> QueryAnswer:
        """
        Queries the knowledge base, answers grounded in documentation,
        and logs a ticket if confidence is insufficient.
        """
        # Sanitize category filter
        if category_filter and category_filter.strip().lower() in ("string", "none", "null", ""):
            category_filter = None

        # 1. Retrieve candidate chunks
        docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            min_score=self.min_confidence_threshold,
            category_filter=category_filter,
        )

        # 2. Check if evidence is insufficient
        if not docs:
            logger.info(f"Insufficient evidence for query: '{question}'. Creating expert ticket.")
            ticket = self.ticket_manager.create_ticket(
                query=question,
                reason="no_relevant_documentation_found",
                context_attempted=[],
                category=category_filter,
            )
            return QueryAnswer(
                query=question,
                answer=(
                    "I don't know based on the provided documentation. "
                    "I have logged a ticket for a human expert to review and provide verified guidance."
                ),
                has_answer=False,
                confidence=0.0,
                citations=[],
                screenshots=[],
                ticket_id=ticket.ticket_id,
            )

        top_score = docs[0].score
        citations = self.retriever.get_unique_citations(docs)
        all_screenshots: List[str] = []
        for d in docs:
            for s in d.screenshot_urls:
                if s and s not in all_screenshots:
                    all_screenshots.append(s)

        # 3. Generate answer
        context_block = self.retriever.format_context_for_prompt(docs)
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Documentation Context:\n{context_block}\n\n"
            f"Please answer the user's question accurately based on the context above."
        )

        if self.api_key:
            try:
                raw_answer = self._call_claude(user_prompt)
            except Exception as e:
                logger.error(f"Error calling Claude API: {e}. Falling back to grounded generator.")
                raw_answer = self._deterministic_grounded_answer(question, docs)
        else:
            raw_answer = self._deterministic_grounded_answer(question, docs)

        # 4. Check if LLM answered "I don't know"
        lower_answer = raw_answer.lower()
        if "i don't know" in lower_answer or "i do not know" in lower_answer:
            ticket = self.ticket_manager.create_ticket(
                query=question,
                reason="model_stated_unknown",
                context_attempted=[d.title for d in docs],
                category=category_filter,
            )
            return QueryAnswer(
                query=question,
                answer=(
                    "I don't know based on the provided documentation. "
                    "A human expert task has been created to answer this question."
                ),
                has_answer=False,
                confidence=top_score,
                citations=citations,
                screenshots=all_screenshots[:3],
                ticket_id=ticket.ticket_id,
            )

        return QueryAnswer(
            query=question,
            answer=raw_answer,
            has_answer=True,
            confidence=top_score,
            citations=citations,
            screenshots=all_screenshots[:4],
            ticket_id=None,
        )
