"""
Knowledge Base subpackage for Task 1: Knowledge Brain.
Contains hierarchical chunker, ChromaDB vector store, knowledge retriever, and query engine.
"""

from .chunker import HierarchyAwareChunker, DocumentChunk
from .vector_store import BDVectorStore, SearchResult
from .retriever import KnowledgeRetriever, RetrievedDocument
from .query_engine import BDQueryEngine, QueryAnswer

__all__ = [
    "HierarchyAwareChunker",
    "DocumentChunk",
    "BDVectorStore",
    "SearchResult",
    "KnowledgeRetriever",
    "RetrievedDocument",
    "BDQueryEngine",
    "QueryAnswer",
]
