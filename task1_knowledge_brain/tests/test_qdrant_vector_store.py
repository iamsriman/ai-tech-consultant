"""Isolated local-mode tests for the Qdrant adapter and migration path."""
from __future__ import annotations

from pathlib import Path

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings

from task1_knowledge_brain.human_loop.ticket_manager import TicketManager
from task1_knowledge_brain.knowledge_base.chunker import DocumentChunk
from task1_knowledge_brain.knowledge_base.qdrant_migration import (
    migrate_chroma_to_qdrant,
    read_chroma_rows,
    verify_qdrant_matches_source,
)
from task1_knowledge_brain.knowledge_base.qdrant_vector_store import (
    QdrantVectorStore,
    qdrant_point_id,
)
from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine
from task1_knowledge_brain.knowledge_base.retriever import KnowledgeRetriever
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore


def embed_texts(texts):
    """Deterministic test embedding with a separate direction per topic."""
    vectors = []
    for text in texts:
        lowered = text.lower()
        vector = np.zeros(384, dtype=np.float32)
        if any(word in lowered for word in ("email", "account", "inbox")):
            vector[0] = 1.0
        elif any(word in lowered for word in ("password", "login")):
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        vectors.append(vector)
    return np.asarray(vectors)


def chunk(chunk_id, article_id, title, category, text, screenshot=""):
    return DocumentChunk(
        chunk_id=chunk_id,
        article_id=article_id,
        title=title,
        category=category,
        folder="Support",
        url=f"https://support.example/articles/{article_id}",
        chunk_index=0,
        total_chunks=1,
        text=text,
        has_screenshot=bool(screenshot),
        screenshot_urls=screenshot,
        source_type="help_center",
    )


def make_local_qdrant(path: Path, collection_name="bd_knowledge_base"):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(path))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    for field in ("article_id", "category"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    return client


def test_qdrant_retrieval_filter_citation_screenshot_and_unknown_ticket(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    client = make_local_qdrant(tmp_path / "qdrant")
    store = QdrantVectorStore(client=client, embedding_function=embed_texts)
    store.add_chunks(
        [
            chunk(
                "article-email_c0",
                "article-email",
                "Create and manage email accounts",
                "Emails",
                "Open Email Accounts to add, edit, and manage email accounts.",
                "https://support.example/images/email.png",
            ),
            chunk(
                "article-password_c0",
                "article-password",
                "Reset an account password",
                "Account Settings",
                "Use the password settings to update a member password.",
                "https://support.example/images/password.png",
            ),
        ]
    )
    assert store.count() == 2
    assert store.query("How do I manage email accounts?", category_filter="Emails")[0].chunk_id == "article-email_c0"
    assert store.query("How do I manage email accounts?", category_filter="Missing") == []

    retriever = KnowledgeRetriever(store)
    docs = retriever.retrieve("How do I manage email accounts?", top_k=3, min_score=0.35)
    assert [doc.title for doc in docs] == ["Create and manage email accounts"]
    assert docs[0].screenshot_urls == ["https://support.example/images/email.png"]
    citations = retriever.get_unique_citations(docs + docs)
    assert len(citations) == 1
    assert citations[0]["url"] == "https://support.example/articles/article-email"

    engine = BDQueryEngine(retriever=retriever, ticket_manager=TicketManager(tmp_path / "tickets.json"))
    unsupported = engine.query("How do I launch a satellite into Mars orbit?")
    assert unsupported.has_answer is False
    assert unsupported.citations == []
    assert unsupported.screenshots == []
    assert unsupported.ticket_id
    assert engine.ticket_manager.get_ticket(unsupported.ticket_id).query == "How do I launch a satellite into Mars orbit?"
    client.close()


def test_qdrant_collection_persists_after_client_restart(tmp_path):
    path = tmp_path / "persistent-qdrant"
    first_client = make_local_qdrant(path)
    first_store = QdrantVectorStore(client=first_client, embedding_function=embed_texts)
    first_store.add_chunks(
        [chunk("persist_c0", "persist", "Email Accounts", "Emails", "Manage email accounts here.")]
    )
    first_client.close()

    from qdrant_client import QdrantClient

    second_client = QdrantClient(path=str(path))
    second_store = QdrantVectorStore(client=second_client, embedding_function=embed_texts)
    assert second_store.count() == 1
    found = second_store.query("manage email accounts")
    assert found[0].chunk_id == "persist_c0"
    assert found[0].title == "Email Accounts"
    second_client.close()


def test_runtime_does_not_create_missing_qdrant_collection(tmp_path):
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="does not exist"):
        QdrantVectorStore(client=client, embedding_function=embed_texts)
    assert client.get_collections().collections == []
    client.close()


def test_selected_qdrant_backend_never_falls_back_to_empty_chroma(tmp_path, monkeypatch):
    chroma_path = tmp_path / "must-not-be-created"
    monkeypatch.setenv("TASK1_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY", "")
    with pytest.raises(RuntimeError, match="QDRANT_URL is missing"):
        BDVectorStore(db_path=chroma_path)
    assert not chroma_path.exists()


def test_qdrant_article_updates_and_deletion_use_original_chunk_ids(tmp_path):
    client = make_local_qdrant(tmp_path / "article-ops")
    store = QdrantVectorStore(client=client, embedding_function=embed_texts)
    store.add_chunks(
        [
            chunk("article-1_c0", "article-1", "Old title", "Emails", "Old email setup details."),
            chunk("article-1_c1", "article-1", "Old title", "Emails", "Old email forwarding details."),
        ]
    )
    assert set(store.get_chunk_ids_for_article("article-1")) == {"article-1_c0", "article-1_c1"}
    store.add_chunks(
        [chunk("article-1_c0", "article-1", "New title", "Emails", "Updated email settings details.")]
    )
    stale = [cid for cid in store.get_chunk_ids_for_article("article-1") if cid != "article-1_c0"]
    assert stale == ["article-1_c1"]
    assert store.delete_chunk_ids(stale) == 1
    assert store.count() == 1
    assert store.delete_article("article-1") == 1
    assert store.count() == 0
    client.close()


def test_chroma_migration_preserves_ids_vectors_metadata_and_is_retryable(tmp_path):
    source_path = tmp_path / "source-chroma"
    chroma = chromadb.PersistentClient(
        path=str(source_path), settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma.create_collection(
        "bd_knowledge_base", metadata={"hnsw:space": "cosine"}
    )
    source_ids = ["1200001_c0", "1200002_c0"]
    source_vectors = np.zeros((2, 384), dtype=np.float32)
    source_vectors[0, 0] = 1.0
    source_vectors[1, 1] = 1.0
    source_docs = ["Email account setup steps.", "Password reset steps."]
    source_metadata = [
        {"chunk_id": source_ids[0], "article_id": "1200001", "title": "Email accounts", "category": "Emails", "url": "https://support.example/email", "has_screenshot": True, "screenshot_urls": "https://support.example/email.png", "source_type": "help_center"},
        {"chunk_id": source_ids[1], "article_id": "1200002", "title": "Password reset", "category": "Account", "url": "https://support.example/password", "has_screenshot": False, "screenshot_urls": "", "source_type": "help_center"},
    ]
    collection.upsert(
        ids=source_ids,
        embeddings=source_vectors,
        documents=source_docs,
        metadatas=source_metadata,
    )

    from qdrant_client import QdrantClient

    qdrant = QdrantClient(path=str(tmp_path / "target-qdrant"))
    report = migrate_chroma_to_qdrant(source_path, qdrant, batch_size=1)
    assert report == {"source_count": 2, "target_count": 2, "verified_points": 2, "uploaded_points": 2, "resumed_points": 0}
    rows = read_chroma_rows(source_path)
    assert verify_qdrant_matches_source(rows, qdrant) == {"source_count": 2, "target_count": 2, "verified_points": 2}
    assert qdrant.retrieve(
        collection_name="bd_knowledge_base",
        ids=[qdrant_point_id(source_ids[0])],
        with_payload=True,
        with_vectors=True,
    )[0].payload["screenshot_urls"] == "https://support.example/email.png"

    second = migrate_chroma_to_qdrant(source_path, qdrant, batch_size=1)
    assert second["uploaded_points"] == 0 and second["resumed_points"] == 2
    assert collection.count() == 2
    assert collection.get(ids=source_ids, include=["embeddings", "documents", "metadatas"])["documents"] == source_docs
    qdrant.close()
    chroma.clear_system_cache()


def test_migration_rejects_unrelated_existing_target_without_overwriting(tmp_path):
    source_path = tmp_path / "source"
    chroma = chromadb.PersistentClient(path=str(source_path), settings=Settings(anonymized_telemetry=False))
    source = chroma.create_collection("bd_knowledge_base", metadata={"hnsw:space": "cosine"})
    source_vector = np.zeros((1, 384), dtype=np.float32)
    source_vector[0, 0] = 1.0
    source.upsert(
        ids=["expected_c0"],
        embeddings=source_vector,
        documents=["Expected source document"],
        metadatas=[{"chunk_id": "expected_c0", "article_id": "expected", "category": "Emails"}],
    )
    import uuid
    from qdrant_client import QdrantClient, models

    qdrant = make_local_qdrant(tmp_path / "target")
    foreign_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name="bd_knowledge_base",
        points=[models.PointStruct(id=foreign_id, vector=[0.0] * 384, payload={"title": "Keep me"})],
        wait=True,
    )
    with pytest.raises(RuntimeError, match="unrelated"):
        migrate_chroma_to_qdrant(source_path, qdrant)
    assert qdrant.count(collection_name="bd_knowledge_base", exact=True).count == 1
    assert qdrant.retrieve(collection_name="bd_knowledge_base", ids=[foreign_id])[0].payload["title"] == "Keep me"
    qdrant.close()
    chroma.clear_system_cache()
