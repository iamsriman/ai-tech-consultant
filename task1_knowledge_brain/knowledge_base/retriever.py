"""
Knowledge Retriever for Task 1: Knowledge Brain.
Retrieves relevant documentation chunks with similarity scoring,
generates citation metadata, and formats context with screenshot references for LLMs.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .vector_store import BDVectorStore, SearchResult


@dataclass
class RetrievedDocument:
    article_id: str
    title: str
    category: str
    folder: str
    url: str
    text: str
    score: float
    screenshot_urls: List[str]
    source_type: str = "legacy_unknown"

    def to_citation(self) -> Dict[str, str | float]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "category": self.category,
            "folder": self.folder,
            "url": self.url,
            "relevance_score": self.score,
        }


class KnowledgeRetriever:
    """
    High-level query retriever supporting similarity thresholds,
    structured citation extraction, and formatted LLM context blocks.
    """

    def __init__(self, vector_store: Optional[BDVectorStore] = None):
        self.vector_store = vector_store or BDVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
        category_filter: Optional[str] = None,
        article_id_filter: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """
        Retrieves matching chunks above the minimum relevance score threshold.
        """
        raw_results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
            category_filter=category_filter,
            article_id_filter=article_id_filter,
        )

        retrieved: List[RetrievedDocument] = []
        for r in raw_results:
            if r.score < min_score:
                continue

            # Split screenshot URLs string into list
            screenshots = [u.strip() for u in r.screenshot_urls.split(";") if u.strip()]

            retrieved.append(
                RetrievedDocument(
                    article_id=r.article_id,
                    title=r.title,
                    category=r.category,
                    folder=r.folder,
                    url=r.url,
                    text=r.text,
                    score=r.score,
                    screenshot_urls=screenshots,
                    source_type=getattr(r, "source_type", "legacy_unknown"),
                )
            )

        return retrieved

    def format_context_for_prompt(self, docs: List[RetrievedDocument]) -> str:
        """
        Formats retrieved chunks into a clean, source-attributed context block
        for the chatbot, inbox agent (Task 2), or scope agent (Task 3).
        """
        if not docs:
            return "No matching help center documentation found."

        formatted_blocks: List[str] = []
        for idx, doc in enumerate(docs, 1):
            block = (
                f"--- SOURCE [{idx}] ---\n"
                f"Title: {doc.title}\n"
                f"Location: {doc.category} > {doc.folder}\n"
                f"URL: {doc.url}\n"
                f"Relevance: {doc.score:.2f}\n"
                f"Content:\n{doc.text}\n"
            )
            if doc.screenshot_urls:
                block += f"Screenshots Available: {', '.join(doc.screenshot_urls[:3])}\n"
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    @staticmethod
    def get_unique_citations(docs: List[RetrievedDocument]) -> List[Dict]:
        """Returns unique articles cited in the retrieved documents."""
        seen_articles = set()
        citations = []
        for doc in docs:
            if doc.article_id not in seen_articles:
                seen_articles.add(doc.article_id)
                citations.append(doc.to_citation())
        return citations
