"""
Pydantic schemas for Task 1: Knowledge Brain FastAPI API.
Provides typed schemas for queries, citations, tickets, settings checking, and sync.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., description="User or agent natural language question", min_length=3)
    top_k: int = Field(default=4, ge=1, le=10, description="Number of source chunks to retrieve")
    category_filter: Optional[str] = Field(default=None, description="Optional category filter (e.g. 'Emails')")
    provider: Optional[str] = Field(default=None, description="Optional provider override: claude or groq")
    model: Optional[str] = Field(default=None, description="Optional model override for the selected provider")

    @field_validator("query", mode="before")
    @classmethod
    def strip_and_require_query(cls, value):
        if not isinstance(value, str):
            return value
        value = value.strip()
        if len(value) < 3:
            raise ValueError("query must contain at least 3 non-whitespace characters")
        return value


class CitationItem(BaseModel):
    article_id: str
    title: str
    category: str
    folder: str
    url: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    has_answer: bool
    confidence: float
    citations: List[CitationItem] = []
    screenshots: List[str] = []
    ticket_id: Optional[str] = None
    model_used: str = "none"
    fallback_used: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    generation_error: Optional[str] = None


class SettingsCheckRequest(BaseModel):
    feature_description: str = Field(..., description="Feature or requirement to check against BD native settings")
    provider: Optional[str] = Field(default=None, description="Optional provider override: claude or groq")
    model: Optional[str] = Field(default=None, description="Optional model override for the selected provider")


class SettingsCheckResponse(BaseModel):
    can_solve_with_native_settings: bool
    matched_setting: Optional[str] = None
    article_title: Optional[str] = None
    article_url: Optional[str] = None
    explanation: str
    screenshots: List[str] = []
    confidence: float


class TicketItem(BaseModel):
    ticket_id: str
    query: str
    reason: str
    status: str
    created_at: str
    resolved_at: Optional[str] = None
    expert_answer: Optional[str] = None
    expert_name: Optional[str] = None
    category: Optional[str] = None


class ResolveTicketRequest(BaseModel):
    expert_answer: str = Field(..., min_length=10, description="The verified correct answer from the expert")
    expert_name: str = Field(default="Senior Technical Consultant", description="Name of the answering consultant")
    category: Optional[str] = Field(default="Expert Solutions", description="Knowledge category")


class ResolveTicketResponse(BaseModel):
    status: str
    ticket_id: str
    chunk_id: str
    indexed_chunks: int
    query: str
    expert_name: str


class SyncRequest(BaseModel):
    max_updates: Optional[int] = Field(default=None, description="Max new/updated articles to crawl in this sync")


class SyncResponse(BaseModel):
    status: str
    timestamp: str
    added_count: int
    updated_count: int
    removed_count: int
    message: str
    unchanged_count: int = 0
    failed_count: int = 0


class HealthResponse(BaseModel):
    status: str
    indexed_chunks_count: int
    pending_tickets_count: int
    resolved_tickets_count: int
    chroma_db_healthy: bool
    # Additive field: keep the legacy health key while identifying the active backend.
    vector_store_backend: str = "chroma"
