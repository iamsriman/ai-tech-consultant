"""Qdrant adapter that preserves Task 1's existing vector-store contract."""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from .chunker import DocumentChunk
from .vector_store import COLLECTION_NAME, SearchResult
from task1_knowledge_brain.config import load_project_environment

logger = logging.getLogger(__name__)
VECTOR_DIMENSION = 384
_POINT_NAMESPACE = uuid.UUID("f88a2c76-2d5d-4e0f-a487-8cce20bd75f3")


def qdrant_point_id(chunk_id: str) -> str:
    """Map Chroma's arbitrary string IDs to stable Qdrant UUID point IDs."""
    return str(uuid.uuid5(_POINT_NAMESPACE, str(chunk_id)))


class QdrantVectorStore:
    """Cloud Qdrant implementation of Task 1's vector operations.

    The runtime adapter requires a pre-existing, compatible collection. Only
    the explicit migration utility is allowed to create one.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        embedding_function: Optional[Callable] = None,
    ) -> None:
        load_project_environment()
        self.collection_name = collection_name
        self.embedding_function = embedding_function or DefaultEmbeddingFunction()
        if client is None:
            resolved_url = (url or os.environ.get("QDRANT_URL", "")).strip()
            resolved_key = (api_key or os.environ.get("QDRANT_API_KEY", "")).strip()
            if not resolved_url:
                raise RuntimeError("QDRANT_URL is missing; configure the Qdrant cluster endpoint.")
            if not resolved_key:
                raise RuntimeError("QDRANT_API_KEY is missing; configure the Qdrant API key.")
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("Qdrant backend selected but qdrant-client is not installed.") from exc
            # Do not log URL, key, or the SDK exception, which may echo request details.
            try:
                client = QdrantClient(
                    url=resolved_url,
                    api_key=resolved_key,
                    timeout=30,
                    prefer_grpc=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Could not initialize Qdrant client ({type(exc).__name__}).") from None
        self.client = client
        self._models = self._load_models()
        self.dimension = self._validate_collection()

    @staticmethod
    def _load_models():
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError("Qdrant backend selected but qdrant-client is not installed.") from exc
        return models

    def _validate_collection(self) -> int:
        try:
            if not self.client.collection_exists(self.collection_name):
                raise RuntimeError(
                    f"Qdrant collection '{self.collection_name}' does not exist. "
                    "Run the Chroma-to-Qdrant migration before selecting this backend."
                )
            info = self.client.get_collection(self.collection_name)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not inspect Qdrant collection ({type(exc).__name__}).") from None

        vector_config = info.config.params.vectors
        if isinstance(vector_config, dict):
            if len(vector_config) != 1 or "" not in vector_config:
                raise RuntimeError("Qdrant collection must use one unnamed dense vector configuration.")
            vector_config = vector_config[""]
        dimension = int(vector_config.size)
        distance = vector_config.distance
        if dimension != VECTOR_DIMENSION:
            raise RuntimeError(
                f"Qdrant collection dimension is {dimension}; Task 1 requires {VECTOR_DIMENSION}."
            )
        if str(getattr(distance, "value", distance)).lower() != "cosine":
            raise RuntimeError("Qdrant collection must use cosine distance to match Chroma retrieval.")
        return dimension

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.asarray(self.embedding_function(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Embedding model returned shape {vectors.shape}; expected ({len(texts)}, {self.dimension})."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned non-finite vector values.")
        return vectors

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        if not chunks:
            return 0
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        models = self._models
        added = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self._embed([chunk.text for chunk in batch])
            points = []
            for chunk, vector in zip(batch, vectors):
                payload = dict(chunk.to_metadata())
                payload["_text"] = chunk.text
                points.append(
                    models.PointStruct(
                        id=qdrant_point_id(chunk.chunk_id),
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            added += len(points)
        return added

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        article_id_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        if n_results < 1 or not query_text.strip():
            return []
        filters = []
        if category_filter:
            filters.append(
                self._models.FieldCondition(
                    key="category", match=self._models.MatchValue(value=category_filter)
                )
            )
        if article_id_filter:
            filters.append(
                self._models.FieldCondition(
                    key="article_id", match=self._models.MatchValue(value=str(article_id_filter))
                )
            )
        query_filter = self._models.Filter(must=filters) if filters else None
        vector = self._embed([query_text])[0].tolist()
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=n_results,
                with_payload=True,
                with_vectors=False,
                # With only 1,812 points this gives stable, exact cosine ranking.
                search_params=self._models.SearchParams(exact=True),
            )
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed ({type(exc).__name__}).") from None

        results: List[SearchResult] = []
        for point in result.points:
            meta = point.payload or {}
            score = round(max(0.0, min(1.0, float(point.score))), 4)
            results.append(
                SearchResult(
                    chunk_id=str(meta.get("chunk_id", "")),
                    article_id=str(meta.get("article_id", "")),
                    title=str(meta.get("title", "")),
                    category=str(meta.get("category", "")),
                    folder=str(meta.get("folder", "")),
                    url=str(meta.get("url", "")),
                    text=str(meta.get("_text", "")),
                    score=score,
                    has_screenshot=bool(meta.get("has_screenshot", False)),
                    screenshot_urls=str(meta.get("screenshot_urls", "")),
                    source_type=str(meta.get("source_type", "legacy_unknown")),
                )
            )
        return results

    def count(self) -> int:
        try:
            return int(self.client.count(collection_name=self.collection_name, exact=True).count)
        except Exception as exc:
            raise RuntimeError(f"Could not count Qdrant vectors ({type(exc).__name__}).") from None

    def get_chunk_ids_for_article(self, article_id: str) -> List[str]:
        filt = self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="article_id", match=self._models.MatchValue(value=str(article_id))
                )
            ]
        )
        ids: List[str] = []
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filt,
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            ids.extend(str(point.payload.get("chunk_id", "")) for point in points if point.payload)
            if next_offset is None:
                return ids
            offset = next_offset

    def delete_chunk_ids(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        point_ids = [qdrant_point_id(chunk_id) for chunk_id in chunk_ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.PointIdsList(points=point_ids),
            wait=True,
        )
        return len(point_ids)

    def delete_article(self, article_id: str) -> int:
        filt = self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="article_id", match=self._models.MatchValue(value=str(article_id))
                )
            ]
        )
        before = int(self.client.count(collection_name=self.collection_name, count_filter=filt, exact=True).count)
        if before:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=self._models.FilterSelector(filter=filt),
                wait=True,
            )
        return before
