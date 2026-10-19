"""
ChromaDB Vector Store for Task 1: Knowledge Brain.
Provides persistent storage, vector similarity querying, metadata filtering,
and article deletion for daily sitemap synchronization.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from .chunker import DocumentChunk
from task1_knowledge_brain.config import load_project_environment

logger = logging.getLogger(__name__)

TASK1_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TASK1_ROOT / "data"
# Canonical Task 1 data location; independent of the caller's working dir.
DEFAULT_CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "bd_knowledge_base"


@dataclass
class SearchResult:
    chunk_id: str
    article_id: str
    title: str
    category: str
    folder: str
    url: str
    text: str
    score: float
    has_screenshot: bool
    screenshot_urls: str
    source_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "category": self.category,
            "folder": self.folder,
            "url": self.url,
            "text": self.text,
            "score": self.score,
            "has_screenshot": self.has_screenshot,
            "screenshot_urls": self.screenshot_urls,
            "source_type": self.source_type,
        }


class BDVectorStore:
    """
    Manages persistent ChromaDB vector storage for Brilliant Directories knowledge base.
    """
    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialize the vector store.

        Args:
            db_path: Optional path to the ChromaDB directory. If omitted,
                defaults to the canonical location ``DEFAULT_CHROMA_DIR``.
        """
        # Load repository settings before choosing a backend. Explicit process
        # environment values still take precedence over .env.
        load_project_environment()
        backend = os.environ.get("TASK1_VECTOR_BACKEND", "chroma").strip().lower()
        if backend not in {"chroma", "qdrant"}:
            raise ValueError("TASK1_VECTOR_BACKEND must be either 'chroma' or 'qdrant'.")
        self.backend_name = backend
        self._qdrant_store = None
        if backend == "qdrant":
            # Qdrant runtime initialization validates an existing collection;
            # it never creates an empty remote collection as a fallback.
            from .qdrant_vector_store import QdrantVectorStore

            self._qdrant_store = QdrantVectorStore()
            self.client = self._qdrant_store.client
            self.collection = None
            return

        storage_path = Path(db_path) if db_path else DEFAULT_CHROMA_DIR
        storage_path.mkdir(parents=True, exist_ok=True)
        # Chroma 1.x selects its SQLite/Rust persistence backend from the path.
        # ``chroma_db_impl=duckdb+parquet`` and ``persist_directory`` are the
        # removed legacy configuration and make current Chroma raise the
        # deprecated-configuration error.  Keep the canonical path unchanged.
        self.client = chromadb.PersistentClient(
            path=storage_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # Retrieve or create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )



    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 1) -> int:
        """
        Adds or updates document chunks in ChromaDB.
        """
        if not chunks:
            return 0

        if self._qdrant_store is not None:
            return self._qdrant_store.add_chunks(chunks, batch_size=batch_size)

        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [c.to_metadata() for c in batch]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.info(f"Upserted {total_added} chunks into {COLLECTION_NAME}.")
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        article_id_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Queries the vector store for chunks matching the query string.
        Supports optional metadata filtering by category or article_id.
        """
        where_filter: Optional[Dict[str, Any]] = None
        if category_filter and article_id_filter:
            where_filter = {
                "$and": [
                    {"category": category_filter},
                    {"article_id": article_id_filter},
                ]
            }
        elif category_filter:
            where_filter = {"category": category_filter}
        elif article_id_filter:
            where_filter = {"article_id": article_id_filter}

        # Ensure n_results does not exceed collection count
        if n_results < 1 or not query_text.strip():
            return []
        if self._qdrant_store is not None:
            return self._qdrant_store.query(
                query_text=query_text,
                n_results=n_results,
                category_filter=category_filter,
                article_id_filter=article_id_filter,
            )
        count = self.collection.count()
        if count == 0:
            return []
        effective_n = min(n_results, count)

        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": effective_n,
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self.collection.query(**kwargs)

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(ids)

        for chunk_id, doc, meta, dist in zip(ids, docs, metadatas, distances):
            # Chroma cosine distance is 1 - cosine_similarity.
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4) if dist is not None else 0.0
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    article_id=str(meta.get("article_id", "")),
                    title=str(meta.get("title", "")),
                    category=str(meta.get("category", "")),
                    folder=str(meta.get("folder", "")),
                    url=str(meta.get("url", "")),
                    text=doc,
                    score=similarity,
                    has_screenshot=bool(meta.get("has_screenshot", False)),
                    screenshot_urls=str(meta.get("screenshot_urls", "")),
                    source_type=str(meta.get("source_type") or (
                        "help_center" if str(meta.get("url", "")).startswith("https://support.brilliantdirectories.com/support/solutions/articles/") else "legacy_unknown"
                    )),
                )
            )

        return search_results

    def delete_article(self, article_id: str) -> int:
        """
        Deletes all chunks belonging to an article ID.
        Used when an article is removed from the documentation sitemap.
        """
        if self._qdrant_store is not None:
            return self._qdrant_store.delete_article(article_id)
        try:
            self.collection.delete(where={"article_id": str(article_id)})
            logger.info(f"Deleted chunks for article_id={article_id} from vector store.")
            return 1
        except Exception as e:
            logger.error(f"Failed to delete article_id {article_id}: {e}")
            return 0

    def count(self) -> int:
        """Returns total number of chunks stored."""
        if self._qdrant_store is not None:
            return self._qdrant_store.count()
        return self.collection.count()

    def get_chunk_ids_for_article(self, article_id: str) -> List[str]:
        """Return original chunk IDs for an article, independent of backend."""
        if self._qdrant_store is not None:
            return self._qdrant_store.get_chunk_ids_for_article(article_id)
        result = self.collection.get(where={"article_id": str(article_id)}, include=[])
        return list(result.get("ids", []))

    def delete_chunk_ids(self, chunk_ids: List[str]) -> int:
        """Delete original chunk IDs, independent of backend."""
        if not chunk_ids:
            return 0
        if self._qdrant_store is not None:
            return self._qdrant_store.delete_chunk_ids(chunk_ids)
        self.collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def reset(self) -> None:
        """Clears all vectors in the collection."""
        if self._qdrant_store is not None:
            raise RuntimeError("Reset is disabled for the remote Qdrant backend to protect stored knowledge.")
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
