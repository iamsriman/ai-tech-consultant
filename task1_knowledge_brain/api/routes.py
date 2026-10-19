"""
FastAPI Router for Task 1: Knowledge Brain.
Exposes endpoints for querying RAG, checking product settings, managing fallback tickets,
ingesting expert solutions, and triggering daily sitemap synchronization.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from .schemas import (
    QueryRequest,
    QueryResponse,
    CitationItem,
    SettingsCheckRequest,
    SettingsCheckResponse,
    TicketItem,
    ResolveTicketRequest,
    ResolveTicketResponse,
    SyncRequest,
    SyncResponse,
    HealthResponse,
)
from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager
from task1_knowledge_brain.human_loop.feedback_ingest import FeedbackIngester
from task1_knowledge_brain.syncer.daily_sync import DailySitemapSyncer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Knowledge Brain"])

# Shared service singletons
vector_store = BDVectorStore()
ticket_manager = TicketManager()
query_engine = BDQueryEngine(ticket_manager=ticket_manager)
feedback_ingester = FeedbackIngester(vector_store=vector_store, ticket_manager=ticket_manager)
daily_syncer = DailySitemapSyncer(vector_store=vector_store)


@router.post("/query", response_model=QueryResponse, summary="Query Knowledge Brain (Chatbot & Task 2)")
def query_knowledge_base(request: QueryRequest):
    """
    Retrieves facts from Brilliant Directories documentation.
    Answers with source citations and screenshot links.
    Automatically creates a human expert ticket if evidence is insufficient.
    """
    # Clean Swagger default placeholder 'string' or empty values
    category = request.category_filter
    if category and category.strip().lower() in ("string", "none", "null", ""):
        category = None

    res = query_engine.query(
        question=request.query,
        top_k=request.top_k,
        category_filter=category,
        provider_name=request.provider,
        model_name=request.model,
    )
    citations = [
        CitationItem(
            article_id=c["article_id"],
            title=c["title"],
            category=c["category"],
            folder=c["folder"],
            url=c["url"],
            relevance_score=c["relevance_score"],
        )
        for c in res.citations
    ]
    return QueryResponse(
        query=res.query,
        answer=res.answer,
        has_answer=res.has_answer,
        confidence=res.confidence,
        citations=citations,
        screenshots=res.screenshots,
        ticket_id=res.ticket_id,
        model_used=res.model_used,
        fallback_used=res.fallback_used,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        generation_error=res.generation_error,
    )


@router.post("/settings/check", response_model=SettingsCheckResponse, summary="Check Native Settings (Task 3)")
def check_native_settings(request: SettingsCheckRequest):
    """
    Specialized endpoint for Task 3 (Scoping Agent).
    Checks if a client feature request can be fulfilled with an existing
    Brilliant Directories setting instead of writing custom code.
    """
    res = query_engine.query(
        question=f"Can this be configured in settings: {request.feature_description}",
        top_k=3,
        provider_name=request.provider,
        model_name=request.model,
    )
    if res.has_answer:
        top_cit = res.citations[0] if res.citations else None
        return SettingsCheckResponse(
            can_solve_with_native_settings=True,
            matched_setting=top_cit["title"] if top_cit else "Native Setting Available",
            article_title=top_cit["title"] if top_cit else None,
            article_url=top_cit["url"] if top_cit else None,
            explanation=res.answer,
            screenshots=res.screenshots,
            confidence=res.confidence,
        )
    return SettingsCheckResponse(
        can_solve_with_native_settings=False,
        matched_setting=None,
        article_title=None,
        article_url=None,
        explanation="No native Brilliant Directories setting found. Likely requires custom development.",
        screenshots=[],
        confidence=res.confidence,
    )


@router.get("/tickets", response_model=List[TicketItem], summary="List Expert Fallback Tickets")
def list_tickets(status_filter: Optional[str] = Query(default=None, alias="status")):
    """Lists unanswered queries requiring human expert review."""
    tickets = ticket_manager.list_tickets(status=status_filter)
    return [TicketItem(**t.to_dict()) for t in tickets]


@router.get("/tickets/{ticket_id}", response_model=TicketItem, summary="Get Ticket Details")
def get_ticket(ticket_id: str):
    """Retrieves details of a specific ticket."""
    ticket = ticket_manager.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return TicketItem(**ticket.to_dict())


@router.post("/tickets/{ticket_id}/resolve", response_model=ResolveTicketResponse, summary="Resolve & Index Expert Answer")
def resolve_ticket(ticket_id: str, request: ResolveTicketRequest):
    """
    Submits a verified human expert answer.
    Marks the ticket as resolved and indexes the answer into ChromaDB memory.
    """
    try:
        result = feedback_ingester.ingest_expert_answer(
            ticket_id=ticket_id,
            expert_answer=request.expert_answer,
            expert_name=request.expert_name,
            category=request.category or "Expert Solutions",
        )
        return ResolveTicketResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error resolving ticket {ticket_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/sync", response_model=SyncResponse, summary="Trigger Daily Sitemap Synchronization")
def trigger_daily_sync(request: SyncRequest = SyncRequest()):
    """
    Performs daily differential sync against support.brilliantdirectories.com/support/sitemap.xml.
    Detects added, updated, and removed articles.
    Safe: Will never delete data if sitemap fetching fails.
    """
    report = daily_syncer.sync(max_updates=request.max_updates)
    return SyncResponse(
        status=report.status,
        timestamp=report.timestamp,
        added_count=len(report.added_articles),
        updated_count=len(report.updated_articles),
        removed_count=len(report.removed_articles),
        message=report.message,
        unchanged_count=report.unchanged_count,
        failed_count=len(report.failed_articles or []),
    )


@router.get("/health", response_model=HealthResponse, summary="Health & Status Check")
def health_check():
    """Returns system status, indexed chunk count, and ticket statistics."""
    try:
        count = vector_store.count()
        db_healthy = True
    except Exception:
        logger.exception("Chroma health check failed")
        count = 0
        db_healthy = False
    pending = ticket_manager.count(status="pending")
    resolved = ticket_manager.count(status="resolved")
    return HealthResponse(
        status="ok" if db_healthy else "degraded",
        indexed_chunks_count=count,
        pending_tickets_count=pending,
        resolved_tickets_count=resolved,
        chroma_db_healthy=db_healthy,
        vector_store_backend=getattr(vector_store, "backend_name", "chroma"),
    )
