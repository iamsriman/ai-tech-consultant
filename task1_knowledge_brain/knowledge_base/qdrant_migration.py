"""Non-destructive transfer and verification helpers for Chroma to Qdrant."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings

from .qdrant_vector_store import VECTOR_DIMENSION, qdrant_point_id
from .vector_store import COLLECTION_NAME, DEFAULT_CHROMA_DIR
from task1_knowledge_brain.config import load_project_environment


def read_chroma_rows(source_path: Path, collection_name: str = COLLECTION_NAME) -> List[Dict[str, Any]]:
    """Read vectors, documents and metadata without recalculating embeddings."""
    source_path = Path(source_path).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Chroma source directory does not exist: {source_path}")
    client = chromadb.PersistentClient(
        path=str(source_path),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(f"Could not open Chroma collection ({type(exc).__name__}).") from None
    if collection.metadata and collection.metadata.get("hnsw:space", "cosine") != "cosine":
        raise RuntimeError("The Chroma source collection is not configured for cosine similarity.")
    count = int(collection.count())
    if count <= 0:
        raise RuntimeError("The Chroma source collection is empty.")
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = result.get("ids", [])
    embeddings = np.asarray(result.get("embeddings"), dtype=np.float32)
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    if len(ids) != count or len(set(ids)) != count:
        raise RuntimeError("Chroma source has missing or duplicate chunk IDs.")
    if embeddings.shape != (count, VECTOR_DIMENSION) or not np.isfinite(embeddings).all():
        raise RuntimeError(
            f"Chroma vectors have shape {embeddings.shape}; expected ({count}, {VECTOR_DIMENSION}) finite vectors."
        )
    if len(documents) != count or any(not isinstance(doc, str) or not doc for doc in documents):
        raise RuntimeError("Chroma source has missing or empty document text.")
    if len(metadatas) != count or any(not isinstance(meta, dict) for meta in metadatas):
        raise RuntimeError("Chroma source has missing or malformed metadata.")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0):
        raise RuntimeError(
            "Qdrant cosine storage normalizes vectors; source vectors are not unit length, refusing to alter them."
        )
    return [
        {
            "chunk_id": str(chunk_id),
            "vector": vector.tolist(),
            "document": document,
            "metadata": dict(metadata),
        }
        for chunk_id, vector, document, metadata in zip(ids, embeddings, documents, metadatas)
    ]


def _models():
    try:
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError("Install qdrant-client before using the migration utility.") from exc
    return models


def _collection_info(client: Any, collection_name: str):
    models = _models()
    info = client.get_collection(collection_name)
    config = info.config.params.vectors
    if isinstance(config, dict):
        if len(config) != 1 or "" not in config:
            raise RuntimeError("Target collection must use one unnamed dense vector configuration.")
        config = config[""]
    if int(config.size) != VECTOR_DIMENSION:
        raise RuntimeError(
            f"Target Qdrant collection has {config.size} dimensions; source requires {VECTOR_DIMENSION}."
        )
    if str(getattr(config.distance, "value", config.distance)).lower() != "cosine":
        raise RuntimeError("Target Qdrant collection must use cosine distance.")
    return info


def _iter_target_points(client: Any, collection_name: str) -> Iterable[Any]:
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        yield from points
        if next_offset is None:
            break
        offset = next_offset


def _assert_point_matches(point: Any, row: Dict[str, Any]) -> None:
    payload = point.payload or {}
    if payload.get("chunk_id") != row["chunk_id"] or payload.get("_text") != row["document"]:
        raise RuntimeError(f"Existing Qdrant point conflicts with Chroma chunk {row['chunk_id']}.")
    for key, value in row["metadata"].items():
        if payload.get(key) != value:
            raise RuntimeError(f"Existing Qdrant metadata conflicts with Chroma chunk {row['chunk_id']}.")
    if not np.allclose(np.asarray(point.vector, dtype=np.float32), row["vector"], atol=2e-5, rtol=0):
        raise RuntimeError(f"Existing Qdrant vector conflicts with Chroma chunk {row['chunk_id']}.")


def _existing_points(client: Any, collection_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reject foreign/conflicting data before any target writes are attempted."""
    models = _models()
    expected = {qdrant_point_id(row["chunk_id"]): row for row in rows}
    existing: Dict[str, Any] = {}
    for point in _iter_target_points(client, collection_name):
        point_id = str(point.id)
        if point_id not in expected:
            raise RuntimeError("Target Qdrant collection contains points unrelated to this Chroma index.")
        _assert_point_matches(point, expected[point_id])
        existing[point_id] = point
    return existing


def ensure_qdrant_collection(client: Any, collection_name: str, *, create: bool) -> bool:
    """Validate/create cosine collection and indexes; return whether it existed."""
    models = _models()
    existed = bool(client.collection_exists(collection_name))
    if existed:
        info = _collection_info(client, collection_name)
        schema = getattr(info, "payload_schema", None) or {}
    elif create:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.COSINE),
        )
        schema = {}
    else:
        return False
    for field in ("article_id", "category"):
        if field not in schema:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
    return existed


def verify_qdrant_matches_source(
    source_rows: List[Dict[str, Any]], client: Any, collection_name: str = COLLECTION_NAME
) -> Dict[str, int]:
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Qdrant collection '{collection_name}' is not present.")
    _collection_info(client, collection_name)
    count = int(client.count(collection_name=collection_name, exact=True).count)
    if count != len(source_rows):
        raise RuntimeError(f"Qdrant count is {count}; expected {len(source_rows)} Chroma chunks.")
    existing = _existing_points(client, collection_name, source_rows)
    if len(existing) != len(source_rows):
        raise RuntimeError("Qdrant does not contain every expected Chroma chunk ID.")
    return {"source_count": len(source_rows), "target_count": count, "verified_points": len(existing)}


def migrate_chroma_to_qdrant(
    source_path: Path,
    client: Any,
    *,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 128,
    dry_run: bool = False,
) -> Dict[str, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = read_chroma_rows(source_path, collection_name)
    existed = bool(client.collection_exists(collection_name))
    if existed:
        _collection_info(client, collection_name)
        before = _existing_points(client, collection_name, rows)
    else:
        before = {}
    if dry_run:
        return {
            "source_count": len(rows),
            "target_count_before": int(client.count(collection_name=collection_name, exact=True).count) if existed else 0,
            "points_to_upload": len(rows) - len(before),
            "collection_exists": int(existed),
        }

    ensure_qdrant_collection(client, collection_name, create=True)
    models = _models()
    pending = [row for row in rows if qdrant_point_id(row["chunk_id"]) not in before]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        points = [
            models.PointStruct(
                id=qdrant_point_id(row["chunk_id"]),
                vector=row["vector"],
                payload={**row["metadata"], "chunk_id": row["chunk_id"], "_text": row["document"]},
            )
            for row in batch
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)

    verification = verify_qdrant_matches_source(rows, client, collection_name)
    return {
        **verification,
        "uploaded_points": len(pending),
        "resumed_points": len(before),
    }


def _make_client():
    load_project_environment()
    url = os.environ.get("QDRANT_URL", "").strip()
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not url:
        raise RuntimeError("QDRANT_URL is missing; add the cluster endpoint to the local .env file.")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY is missing; add the key to the local .env file.")
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=url, api_key=api_key, timeout=60, prefer_grpc=False)
    except Exception as exc:
        raise RuntimeError(f"Could not connect to Qdrant ({type(exc).__name__}); credentials were not displayed.") from None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="Validate source and target without writing to Qdrant")
    parser.add_argument("--verify-only", action="store_true", help="Verify a completed migration without writing")
    args = parser.parse_args(argv)
    try:
        rows = read_chroma_rows(args.source_path, args.collection)
        client = _make_client()
        if args.verify_only:
            report = verify_qdrant_matches_source(rows, client, args.collection)
        else:
            report = migrate_chroma_to_qdrant(
                args.source_path,
                client,
                collection_name=args.collection,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
        print(json.dumps({"status": "verified" if args.verify_only else "ok", **report}, sort_keys=True))
        client.close()
        return 0
    except Exception as exc:
        # Avoid printing SDK exception text because HTTP libraries may include request details.
        message = str(exc)
        secret = os.environ.get("QDRANT_API_KEY", "")
        if secret:
            message = message.replace(secret, "[redacted]")
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": message}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
