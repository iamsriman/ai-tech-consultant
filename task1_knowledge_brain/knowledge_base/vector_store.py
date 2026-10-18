"""
ChromaDB Vector Store for Task 1: Knowledge Brain.
Provides persistent storage, vector similarity querying, metadata filtering,
and article deletion for daily sitemap synchronization.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from .chunker import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIR = Path("task1_knowledge_brain/data/chroma_db")
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
        }


class BDVectorStore:
    """
    Manages persistent ChromaDB vector storage for Brilliant Directories knowledge base.
    """

    def __init__(self, persist_directory: Path = DEFAULT_CHROMA_DIR):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as e:
            logger.error(f"Failed to open persistent ChromaDB at {self.persist_directory}: {e}")
            # Fallback to a fresh temporary directory to keep the app running
            fallback_dir = self.persist_directory.parent / "chroma_db_fallback"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using fallback ChromaDB directory at {fallback_dir}")
            self.client = chromadb.PersistentClient(path=str(fallback_dir))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        """
        Adds or updates document chunks in ChromaDB.
        """
        if not chunks:
            return 0

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
            # Chroma with cosine distance returns distance d in [0, 2], similarity score = 1 - (d / 2)
            similarity = round(max(0.0, 1.0 - (dist / 2.0)), 4) if dist is not None else 1.0
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
                )
            )

        return search_results

    def delete_article(self, article_id: str) -> int:
        """
        Deletes all chunks belonging to an article ID.
        Used when an article is removed from the documentation sitemap.
        """
        try:
            self.collection.delete(where={"article_id": str(article_id)})
            logger.info(f"Deleted chunks for article_id={article_id} from vector store.")
            return 1
        except Exception as e:
            logger.error(f"Failed to delete article_id {article_id}: {e}")
            return 0

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Clears all vectors in the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
