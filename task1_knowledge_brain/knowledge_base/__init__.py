"""
Knowledge Base subpackage for Task 1: Knowledge Brain.
Contains hierarchical chunker, ChromaDB vector store, and knowledge retriever.
"""

from .chunker import HierarchyAwareChunker, DocumentChunk
from .vector_store import BDVectorStore, SearchResult
from .retriever import KnowledgeRetriever, RetrievedDocument

__all__ = [
    "HierarchyAwareChunker",
    "DocumentChunk",
    "BDVectorStore",
    "SearchResult",
    "KnowledgeRetriever",
    "RetrievedDocument",
]
