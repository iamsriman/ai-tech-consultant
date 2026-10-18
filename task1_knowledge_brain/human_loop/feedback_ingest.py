"""
Feedback Ingestion Pipeline for Task 1: Knowledge Brain.
Takes approved expert answers from the ticket queue and indexes them into
ChromaDB so future queries immediately benefit from the newly learned knowledge.
"""

import logging
from typing import Dict, Optional
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
from task1_knowledge_brain.knowledge_base.chunker import DocumentChunk
from .ticket_manager import TicketManager, ExpertTicket

logger = logging.getLogger(__name__)


class FeedbackIngester:
    """
    Ingests expert human resolutions back into ChromaDB memory.
    """

    def __init__(
        self,
        vector_store: Optional[BDVectorStore] = None,
        ticket_manager: Optional[TicketManager] = None,
    ):
        self.vector_store = vector_store or BDVectorStore()
        self.ticket_manager = ticket_manager or TicketManager()

    def ingest_expert_answer(
        self,
        ticket_id: str,
        expert_answer: str,
        expert_name: str = "Senior Support Consultant",
        category: str = "Expert Solutions",
        folder: str = "Verified Answers",
    ) -> Dict[str, str | int]:
        """
        Resolves the ticket and indexes the expert Q&A into ChromaDB vector memory.
        """
        ticket = self.ticket_manager.get_ticket(ticket_id)
        if not ticket:
            raise ValueError(f"Ticket with ID '{ticket_id}' not found.")

        if ticket.status == "resolved":
            raise ValueError(f"Ticket '{ticket_id}' is already resolved")
        if not expert_answer or not expert_answer.strip():
            raise ValueError("Expert answer must not be empty")

        # 2. Format knowledge chunk
        chunk_text = (
            f"[Article: Expert Answer: {ticket.query} | Category: {category} | "
            f"Folder: {folder} | URL: https://support.brilliantdirectories.com/support/expert/{ticket_id}]\n\n"
            f"## Question\n{ticket.query}\n\n"
            f"## Verified Solution\n{expert_answer}\n\n"
            f"[SOURCE_TYPE: verified_expert]\n(Submitted by {expert_name}; ticket {ticket_id})"
        )

        doc_chunk = DocumentChunk(
            chunk_id=f"expert_{ticket_id}_c0",
            article_id=f"expert_{ticket_id}",
            title=f"Expert Answer: {ticket.query}",
            category=category,
            folder=folder,
            url=f"https://support.brilliantdirectories.com/support/expert/{ticket_id}",
            chunk_index=0,
            total_chunks=1,
            text=chunk_text,
            has_screenshot=False,
            screenshot_urls="",
            source_type="verified_expert",
            verified_by=expert_name,
        )

        # 3. Upsert into ChromaDB
        added_count = self.vector_store.add_chunks([doc_chunk])
        if added_count != 1:
            raise RuntimeError("Expert answer was not indexed; ticket remains pending")
        # Persist resolution only after the knowledge write succeeded.
        self.ticket_manager.resolve_ticket(ticket_id=ticket_id, expert_answer=expert_answer, expert_name=expert_name)
        logger.info(
            f"Successfully indexed expert answer for ticket {ticket_id} into ChromaDB ({added_count} chunk)."
        )

        return {
            "status": "success",
            "ticket_id": ticket_id,
            "chunk_id": doc_chunk.chunk_id,
            "indexed_chunks": added_count,
            "query": ticket.query,
            "expert_name": expert_name,
        }
