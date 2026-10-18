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
from task1_knowledge_brain.config import anthropic_model, load_project_environment
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager, ExpertTicket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the AI Knowledge Brain for Brilliant Directories Solution Center.
Your role is to answer questions accurately and honestly based SOLELY on the provided documentation sources.

STRICT RULES:
1. Grounding: Answer strictly using facts from the provided sources. Do NOT make up settings, menus, or features.
2. Honesty: If the documentation does not contain enough information to answer the question, state honestly:
   "I don't know based on the provided documentation."
3. Citations and screenshots are returned separately by the API. Never invent or output citations, URLs, or screenshot links.
4. Tone: Concise, professional, and clear for founders and webmasters.
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
    model_used: str = "none"
    fallback_used: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    generation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "has_answer": self.has_answer,
            "confidence": self.confidence,
            "citations": self.citations,
            "screenshots": self.screenshots,
            "ticket_id": self.ticket_id,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_error": self.generation_error,
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
        model_name: Optional[str] = None,
        min_confidence_threshold: float = 0.35,
    ):
        # Backward compatibility: tests may pass vector_store directly
        if vector_store is not None and retriever is None:
            retriever = KnowledgeRetriever(vector_store)
        self.retriever = retriever or KnowledgeRetriever()
        self.ticket_manager = ticket_manager or TicketManager()
        load_project_environment()
        self.model_name = model_name or anthropic_model()
        self.min_confidence_threshold = min_confidence_threshold
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    def _call_claude(self, prompt: str) -> Dict[str, Any]:
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
            text = "\n".join(block["text"] for block in data.get("content", []) if block.get("type") == "text")
            if not text:
                raise ValueError("Anthropic returned no text content")
            usage = data.get("usage", {})
            return {"text": text, "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"how", "what", "where", "when", "why", "who", "do", "does", "did", "can", "could", "would", "i", "we", "you", "my", "the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "is", "are", "with", "using", "brilliant", "directories", "please", "tell", "about", "custom", "inside", "next", "week", "system", "website", "site", "set", "up", "install", "configure", "access", "using", "bd", "kernel"}
        words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in stop}
        normalized = set()
        for word in words:
            if len(word) > 5 and word.endswith("ies"):
                word = word[:-3] + "y"
            elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]
            normalized.add(word)
        return normalized

    @staticmethod
    def _evidence_text(doc: RetrievedDocument) -> str:
        if doc.source_type == "verified_expert":
            marker = "## Verified Solution\n"
            if marker not in doc.text or "[SOURCE_TYPE: VERIFIED_EXPERT]" not in doc.text.upper():
                return ""
            answer_body = doc.text.split(marker, 1)[1].split("[SOURCE_TYPE:", 1)[0]
            return f"{doc.category} {doc.folder} {answer_body}"
        return f"{doc.title} {doc.category} {doc.folder} {doc.text}"

    def _evidence_overlap(self, question: str, doc: RetrievedDocument) -> int:
        return len(self._terms(question) & self._terms(self._evidence_text(doc)))

    def _title_overlap(self, question: str, doc: RetrievedDocument) -> int:
        return len(self._terms(question) & self._terms(f"{doc.title} {doc.category} {doc.folder}"))

    def _evidence_relevant(self, question: str, doc: RetrievedDocument) -> bool:
        # Similarity ranks candidates; lexical coverage is a separate evidence
        # check and is not interpreted as a calibrated probability.
        question_terms = self._terms(question)
        overlap = self._evidence_overlap(question, doc)
        # Do not let the Q&A label/query embedded in old synthetic expert rows
        # make itself relevant. Expert records require trusted provenance.
        if doc.source_type not in {"help_center", "verified_expert"}:
            return False
        # A lone shared keyword is too weak to establish that evidence answers
        # the question (for example, an unrelated article mentioning one noun).
        return len(question_terms) >= 2 and overlap >= 2

    def _evidence_coverage(self, question: str, docs: List[RetrievedDocument]) -> float:
        """Lexical evidence coverage for the legacy confidence field (not a probability)."""
        terms = self._terms(question)
        if not terms:
            return 0.0
        best = 0
        for doc in docs:
            if doc.source_type == "verified_expert":
                body = doc.text.split("## Verified Solution\n", 1)[-1].split("[SOURCE_TYPE:", 1)[0]
                evidence = f"{doc.category} {doc.folder} {body}"
            else:
                evidence = f"{doc.title} {doc.category} {doc.folder} {doc.text}"
            best = max(best, len(terms & self._terms(evidence)))
        return round(best / len(terms), 4)

    @staticmethod
    def _anthropic_failure_message(error: Exception) -> str:
        if isinstance(error, httpx.TimeoutException):
            return "Anthropic request timed out"
        if isinstance(error, httpx.HTTPStatusError):
            response = error.response
            if response.status_code == 429:
                return "Anthropic rate limit reached"
            if response.status_code == 401:
                return "Anthropic rejected the API credential (HTTP 401)"
            if response.status_code == 403:
                return "Anthropic denied this API request (HTTP 403)"
            if response.status_code >= 500:
                return "Anthropic server error"
            detail = ""
            try:
                payload = response.json().get("error", {})
                detail = f"{payload.get('type', '')} {payload.get('message', '')}".lower()
            except Exception:
                pass
            if response.status_code == 402 or "credit" in detail or "billing" in detail:
                return "Anthropic account has insufficient credits or a billing issue"
            return f"Anthropic rejected the request (HTTP {response.status_code})"
        return f"Anthropic request failed ({type(error).__name__})"

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
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        docs = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            min_score=self.min_confidence_threshold,
            category_filter=category_filter,
        )
        docs = [d for d in docs if self._evidence_relevant(question, d)]
        # Prefer a source that covers more of the question's key terms; use
        # vector distance only to order candidates with equal evidence coverage.
        docs.sort(key=lambda d: (self._evidence_overlap(question, d), self._title_overlap(question, d), d.score), reverse=True)

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
                model_used="deterministic_grounded",
                fallback_used=True,
            )

        evidence_coverage = self._evidence_coverage(question, docs)
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

        model_used = "deterministic_grounded"
        fallback_used = True
        input_tokens = output_tokens = None
        generation_error = None if self.api_key else "ANTHROPIC_API_KEY is not configured"
        if self.api_key:
            try:
                model_result = self._call_claude(user_prompt)
                raw_answer = model_result["text"]
                input_tokens = model_result.get("input_tokens")
                output_tokens = model_result.get("output_tokens")
                model_used = self.model_name
                fallback_used = False
            except Exception as e:
                # Exception text may contain request details; avoid logging secrets.
                logger.error("Anthropic request failed (%s); using explicitly identified grounded fallback.", type(e).__name__)
                generation_error = self._anthropic_failure_message(e)
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
                confidence=evidence_coverage,
                citations=[],
                screenshots=[],
                ticket_id=ticket.ticket_id,
                model_used=model_used,
                fallback_used=fallback_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_error=generation_error,
            )

        if fallback_used:
            # The deterministic fallback answers from its single source block;
            # return only that article's citation and screenshots.
            citations = self.retriever.get_unique_citations(docs[:1])
            all_screenshots = list(docs[0].screenshot_urls) if docs else []

        return QueryAnswer(
            query=question,
            answer=raw_answer,
            has_answer=True,
            confidence=evidence_coverage,
            citations=citations,
            screenshots=all_screenshots[:4],
            ticket_id=None,
            model_used=model_used,
            fallback_used=fallback_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            generation_error=generation_error,
        )
