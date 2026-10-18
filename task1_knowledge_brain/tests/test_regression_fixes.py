from types import SimpleNamespace
import json

from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine
from task1_knowledge_brain.knowledge_base.retriever import RetrievedDocument
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager
from task1_knowledge_brain.human_loop.feedback_ingest import FeedbackIngester
from task1_knowledge_brain.knowledge_base.vector_store import DEFAULT_CHROMA_DIR
from task1_knowledge_brain.knowledge_base.retriever import KnowledgeRetriever
from task1_knowledge_brain.multimodal.image_describer import ImageDescriber
from task1_knowledge_brain.multimodal.image_extractor import ExtractedImage


class StubRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vector_store = SimpleNamespace()

    def retrieve(self, **kwargs):
        return self.docs[: kwargs["top_k"]]

    @staticmethod
    def get_unique_citations(docs):
        return [{"article_id": d.article_id, "title": d.title, "category": d.category,
                 "folder": d.folder, "url": d.url, "relevance_score": d.score} for d in docs]

    @staticmethod
    def format_context_for_prompt(docs):
        return "\n".join(d.text for d in docs)


def doc(*, article_id="5000558363", title="Emails: Email Accounts", category="Emails", folder="Email Accounts",
        text="Create and manage email accounts from the Email Accounts area.", source_type="help_center", screenshot_urls=None):
    return RetrievedDocument(article_id, title, category, folder,
        f"https://support.brilliantdirectories.com/support/solutions/articles/{article_id}-email-accounts",
        text, .83, screenshot_urls or [], source_type)


def test_unrelated_question_is_rejected_and_ticketed(tmp_path):
    tickets = TicketManager(tmp_path / "tickets.json")
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=tickets)
    result = engine.query("How do I put a satellite into Mars orbit?")
    assert result.has_answer is False
    assert result.ticket_id
    assert not result.citations
    assert result.model_used == "deterministic_grounded" and result.fallback_used is True
    assert tickets.get_ticket(result.ticket_id).query == "How do I put a satellite into Mars orbit?"


def test_relevant_email_question_uses_only_relevant_citations(tmp_path):
    relevant = doc(screenshot_urls=["https://images.example/email.png"])
    irrelevant = doc(article_id="12000109949", title="Change Password", category="Dashboard", folder="Account",
                     text="Reset your account password.", screenshot_urls=["https://images.example/password.png"])
    engine = BDQueryEngine(retriever=StubRetriever([relevant, irrelevant]), ticket_manager=TicketManager(tmp_path / "tickets.json"))
    result = engine.query("How do I create and manage email accounts?")
    assert result.has_answer is True
    assert [c["title"] for c in result.citations] == ["Emails: Email Accounts"]
    assert result.screenshots == ["https://images.example/email.png"]
    assert result.model_used == "deterministic_grounded"
    assert result.fallback_used is True


def test_untyped_expert_records_are_not_evidence(tmp_path):
    legacy = doc(title="Expert Answer: satellite", text="Verified answer: use orbital settings.", source_type="legacy_unknown")
    engine = BDQueryEngine(retriever=StubRetriever([legacy]), ticket_manager=TicketManager(tmp_path / "tickets.json"))
    result = engine.query("How do I configure orbital settings?")
    assert result.has_answer is False


def test_single_shared_keyword_is_not_sufficient_evidence(tmp_path):
    engine = BDQueryEngine(retriever=StubRetriever([doc(title="Satellite integration options", text="An unrelated product integration mentions satellites.")]),
                           ticket_manager=TicketManager(tmp_path / "tickets.json"))
    result = engine.query("How do I set up a satellite?")
    assert result.has_answer is False
    assert result.ticket_id is not None


def test_anthropic_usage_is_propagated(tmp_path, monkeypatch):
    import json
    import httpx
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / "tickets.json"))

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"content":[{"type":"text","text":"Use the Email Accounts area."}],"usage":{"input_tokens":71,"output_tokens":19}}
    class FakeClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, headers, json):
            assert url.endswith("/v1/messages")
            assert json["model"] == engine.model_name
            return FakeResponse()
    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = engine.query("How do I create and manage email accounts?")
    assert result.model_used == engine.model_name
    assert result.fallback_used is False
    assert (result.input_tokens, result.output_tokens) == (71, 19)


def test_anthropic_failure_is_identified_as_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / "tickets.json"))
    def fail(prompt): raise RuntimeError("simulated unavailable API")
    engine._call_claude = fail
    result = engine.query("How do I create and manage email accounts?")
    assert result.has_answer
    assert result.model_used == "deterministic_grounded"
    assert result.fallback_used
    assert result.generation_error == "Anthropic request failed (RuntimeError)"


def test_anthropic_credential_rejection_is_clear_and_never_claims_claude(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / "tickets.json"))
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(401, request=request)
    engine._call_claude = lambda prompt: (_ for _ in ()).throw(httpx.HTTPStatusError("unauthorized", request=request, response=response))
    result = engine.query("How do I create and manage email accounts?")
    assert result.model_used == "deterministic_grounded"
    assert result.fallback_used is True
    assert result.input_tokens is None and result.output_tokens is None
    assert result.generation_error == "Anthropic rejected the API credential (HTTP 401)"


def test_anthropic_credit_rate_timeout_and_server_errors_are_explicit(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cases = [(402, "Anthropic account has insufficient credits or a billing issue"),
             (429, "Anthropic rate limit reached"), (503, "Anthropic server error")]
    for status_code, message in cases:
        engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / f"tickets-{status_code}.json"))
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(status_code, request=request, json={"error": {"type": "billing_error", "message": "billing"}})
        engine._call_claude = lambda prompt, req=request, resp=response: (_ for _ in ()).throw(
            httpx.HTTPStatusError("provider failure", request=req, response=resp))
        result = engine.query("How do I create and manage email accounts?")
        assert result.generation_error == message
        assert result.fallback_used is True and result.model_used == "deterministic_grounded"
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / "timeout.json"))
    engine._call_claude = lambda prompt: (_ for _ in ()).throw(httpx.TimeoutException("timeout"))
    assert engine.query("How do I create and manage email accounts?").generation_error == "Anthropic request timed out"


def test_missing_anthropic_key_is_reported_without_provider_call(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=TicketManager(tmp_path / "tickets.json"))
    engine._call_claude = lambda prompt: (_ for _ in ()).throw(AssertionError("must not call provider"))
    result = engine.query("How do I create and manage email accounts?")
    assert result.fallback_used is True and result.model_used == "deterministic_grounded"
    assert result.generation_error == "ANTHROPIC_API_KEY is not configured"
    assert result.input_tokens is None and result.output_tokens is None


def test_explicit_empty_api_key_overrides_dotenv_by_design(monkeypatch):
    from task1_knowledge_brain.config import load_project_environment
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    load_project_environment()
    assert __import__("os").environ["ANTHROPIC_API_KEY"] == ""


def test_expert_answer_is_indexed_before_ticket_resolution(tmp_path):
    manager = TicketManager(tmp_path / "tickets.json")
    ticket = manager.create_ticket("How do I configure email routing?")

    class Store:
        def __init__(self, result): self.result, self.chunks = result, []
        def add_chunks(self, chunks): self.chunks.extend(chunks); return self.result

    store = Store(1)
    result = FeedbackIngester(store, manager).ingest_expert_answer(
        ticket.ticket_id, "Configure email routing from the verified email settings screen.", "Ada Expert"
    )
    assert result["indexed_chunks"] == 1
    assert store.chunks[0].source_type == "verified_expert"
    assert manager.get_ticket(ticket.ticket_id).status == "resolved"
    indexed = store.chunks[0]
    followup = BDQueryEngine(
        retriever=StubRetriever([RetrievedDocument(
            indexed.article_id, indexed.title, indexed.category, indexed.folder, indexed.url,
            indexed.text, .9, [], "verified_expert"
        )]), ticket_manager=manager
    ).query(ticket.query)
    assert followup.has_answer is True
    assert followup.citations[0]["article_id"] == f"expert_{ticket.ticket_id}"
    try:
        FeedbackIngester(store, manager).ingest_expert_answer(
            ticket.ticket_id, "Configure email routing from the verified email settings screen.", "Ada Expert"
        )
        assert False, "duplicate resolution should be rejected"
    except ValueError:
        pass


def test_expert_answer_index_failure_leaves_ticket_pending(tmp_path):
    manager = TicketManager(tmp_path / "tickets.json")
    ticket = manager.create_ticket("How do I configure email routing?")

    class FailingStore:
        def add_chunks(self, chunks): return 0

    try:
        FeedbackIngester(FailingStore(), manager).ingest_expert_answer(
            ticket.ticket_id, "Configure email routing from the verified email settings screen.", "Ada Expert"
        )
        assert False, "index failure should raise"
    except RuntimeError:
        pass
    assert manager.get_ticket(ticket.ticket_id).status == "pending"


def test_image_descriptions_are_cached_and_reused(tmp_path):
    image = ExtractedImage("https://images.example/ui.png", alt="Click the email setup tab and select the account.")
    first = ImageDescriber(tmp_path / "images.json")
    description = first.describe_image(image, "Email Accounts", "Emails", "Email Accounts")
    reloaded_image = {"src": image.src, "alt": image.alt}
    contextual = first.describe_image(reloaded_image, "Other Article", "Other", "Other")
    assert image.description_source == "alt_or_context"
    assert reloaded_image["description_source"] == "alt_or_context"
    assert "Other Article" in contextual
    assert description.startswith("Screenshot in 'Email Accounts'")


def test_image_vision_uses_mocked_model_and_reprocesses_changed_bytes(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    describer = ImageDescriber(tmp_path / "images.json")
    downloads = [b"image version one", b"image version one", b"image version two"]
    model_calls = []
    def download(url, prior):
        body = downloads.pop(0)
        return body, "image/png", {"etag":"test-etag","last_modified":None,"media_type":"image/png"}, False
    def vision(body, media_type, title, category, folder):
        model_calls.append(body)
        return ("Visible admin controls: Email Accounts menu.", {"input_tokens":22,"output_tokens":8})
    monkeypatch.setattr(describer, "_download_image", download)
    monkeypatch.setattr(describer, "_call_vision_model", vision)
    one = ExtractedImage("https://s3.amazonaws.com/screenshot.png")
    desc = describer.describe_image(one, "Email Accounts", "Emails", "Email Accounts")
    assert one.description_source == "claude_vision"
    assert one.content_hash == hashlib.sha256(b"image version one").hexdigest()
    second = ImageDescriber(tmp_path / "images.json")
    monkeypatch.setattr(second, "_download_image", download)
    monkeypatch.setattr(second, "_call_vision_model", vision)
    second.describe_image(ExtractedImage(one.src), "Email Accounts", "Emails", "Email Accounts")
    changed = ExtractedImage(one.src)
    second.describe_image(changed, "Email Accounts", "Emails", "Email Accounts")
    assert changed.content_hash == hashlib.sha256(b"image version two").hexdigest()
    assert len(model_calls) == 2
    assert second.cache[second.get_cache_key(one.src)]["article_title"] == "Email Accounts"


def test_failed_vision_caches_download_without_claiming_visual_analysis(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cache_path = tmp_path / "images.json"
    image = ExtractedImage("https://s3.amazonaws.com/screen.png", alt="Email settings screen")
    first = ImageDescriber(cache_path)
    monkeypatch.setattr(first, "_download_image", lambda url, prior: (
        b"image-bytes", "image/png", {"etag": "v1", "last_modified": None, "media_type": "image/png"}, False
    ))
    monkeypatch.setattr(first, "_call_vision_model", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    first.describe_image(image, "Email Accounts", "Emails", "Email Accounts")
    record = first.cache[first.get_cache_key(image.src)]
    assert record["source"] == "alt_or_context"
    assert record["analysis_error"] == "RuntimeError"
    assert Path(record["local_path"]).read_bytes() == b"image-bytes"

    retry = ImageDescriber(cache_path)
    monkeypatch.setattr(retry, "_download_image", lambda url, prior: (
        Path(prior["local_path"]).read_bytes(), "image/png", prior, True
    ))
    monkeypatch.setattr(retry, "_call_vision_model", lambda *args: ("Visible Email Accounts tab.", {"input_tokens": 8, "output_tokens": 4}))
    retry_image = ExtractedImage(image.src)
    assert retry.describe_image(retry_image, "Email Accounts", "Emails", "Email Accounts") == "Visible Email Accounts tab."
    assert retry_image.description_source == "claude_vision"


def test_duplicate_chunks_collapse_to_one_article_citation():
    duplicate = doc()
    citations = KnowledgeRetriever.get_unique_citations([duplicate, duplicate])
    assert len(citations) == 1


def test_default_database_path_is_absolute_and_canonical():
    assert DEFAULT_CHROMA_DIR.is_absolute()
    assert DEFAULT_CHROMA_DIR == DEFAULT_CHROMA_DIR.parents[0] / "chroma_db"


def test_persisted_expert_cycle_uses_isolated_chroma_and_rejects_unrelated(tmp_path, monkeypatch):
    from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
    from task1_knowledge_brain.knowledge_base.retriever import KnowledgeRetriever
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    store = BDVectorStore(tmp_path / "isolated_chroma")
    manager = TicketManager(tmp_path / "tickets.json")
    ticket = manager.create_ticket("How do I configure custom email forwarding?")
    FeedbackIngester(store, manager).ingest_expert_answer(
        ticket.ticket_id,
        "Open Email Settings, enable custom forwarding, and save the forwarding address.",
        "Verified Support Agent",
        category="Emails",
    )
    reloaded_manager = TicketManager(tmp_path / "tickets.json")
    assert reloaded_manager.get_ticket(ticket.ticket_id).status == "resolved"
    engine = BDQueryEngine(retriever=KnowledgeRetriever(BDVectorStore(tmp_path / "isolated_chroma")),
                           ticket_manager=reloaded_manager)
    relevant = engine.query("How can I configure custom email forwarding?")
    unrelated = engine.query("How do I send a satellite into Mars orbit?")
    assert relevant.has_answer
    assert relevant.citations[0]["article_id"] == f"expert_{ticket.ticket_id}"
    assert not unrelated.has_answer


def test_sync_requires_complete_snapshots_and_two_day_deletion_confirmation(tmp_path):
    from datetime import datetime, timedelta
    from task1_knowledge_brain.syncer.daily_sync import DailySitemapSyncer
    from task1_knowledge_brain.scraper.sitemap_parser import SitemapEntry

    class FakeParser:
        entries = []
        def parse(self): return self.entries
        def get_articles_only(self, entries): return entries

    class FakeCrawler:
        articles_dir = tmp_path / "articles"
        def __init__(self): self.saved = []
        def load_all_saved_articles(self): return self.saved
        def crawl_article(self, entry):
            article = SimpleNamespace(article_id=entry.article_id, lastmod=entry.lastmod,
                                      title="Test", url=entry.loc, to_dict=lambda: {"id": entry.article_id})
            self.saved.append(article)
            return article

    class FakeChunk:
        def chunk_article(self, article):
            return [SimpleNamespace(chunk_id=f"{article.article_id}_c0")]

    class FakeCollection:
        def __init__(self): self.deleted = []
        def get(self, where, include): return {"ids": []}
        def delete(self, where=None, ids=None): self.deleted.append(where or ids)

    class FakeStore:
        def __init__(self): self.collection = FakeCollection(); self.added = []
        def add_chunks(self, chunks): self.added.extend(chunks); return len(chunks)

    parser, crawler, store = FakeParser(), FakeCrawler(), FakeStore()
    syncer = DailySitemapSyncer(parser, crawler, FakeChunk(), store, tmp_path / "snapshot.json")
    parser.entries = [SitemapEntry(f"https://support.brilliantdirectories.com/support/solutions/articles/{i}-a",
                                   article_id=str(i), is_article=True) for i in range(1, 101)]
    first = syncer.sync()
    assert first.status == "success" and len(first.added_articles) == 100
    # A partial sitemap is rejected before any indexed content or snapshot changes.
    before_chunks, before_deletes = len(store.added), len(store.collection.deleted)
    parser.entries = parser.entries[:5]
    partial = syncer.sync()
    assert partial.status == "aborted"
    assert len(store.added) == before_chunks and len(store.collection.deleted) == before_deletes
    # One complete omission records a candidate; a second complete daily snapshot confirms it.
    state = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    state["observed_on"] = (datetime.now().astimezone().date() - timedelta(days=1)).isoformat()
    (tmp_path / "snapshot.json").write_text(json.dumps(state), encoding="utf-8")
    parser.entries = [SitemapEntry(f"https://support.brilliantdirectories.com/support/solutions/articles/{i}-a",
                                   lastmod="2026-10-19" if i == 2 else None,
                                   article_id=str(i), is_article=True) for i in range(2, 102)]
    one_miss = syncer.sync()
    assert one_miss.added_articles == ["101"]
    assert one_miss.updated_articles == ["2"]
    assert one_miss.unchanged_count == 98
    assert one_miss.removed_articles == []
    state = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))
    state["observed_on"] = (datetime.now().astimezone().date() - timedelta(days=1)).isoformat()
    (tmp_path / "snapshot.json").write_text(json.dumps(state), encoding="utf-8")
    confirmed = syncer.sync()
    assert confirmed.removed_articles == ["1"]
    assert {"article_id": "1"} in store.collection.deleted


def test_sync_failed_replacement_preserves_old_indexed_article(tmp_path):
    from task1_knowledge_brain.syncer.daily_sync import DailySitemapSyncer
    from task1_knowledge_brain.scraper.sitemap_parser import SitemapEntry

    class Parser:
        def parse(self):
            return [SitemapEntry(f"https://support.brilliantdirectories.com/support/solutions/articles/{i}-item",
                                 lastmod="new" if i == 1 else None, article_id=str(i), is_article=True)
                    for i in range(1, 101)]
        def get_articles_only(self, entries): return entries

    class Crawler:
        articles_dir = tmp_path / "articles"
        def __init__(self):
            self.saved = [SimpleNamespace(article_id=str(i), url=f"https://support.brilliantdirectories.com/support/solutions/articles/{i}-item",
                                          lastmod="old" if i == 1 else None) for i in range(1, 101)]
        def load_all_saved_articles(self): return self.saved
        def crawl_article(self, entry): return None

    class Collection:
        def __init__(self): self.deletions = []
        def get(self, where, include): return {"ids": ["1_old_chunk"]}
        def delete(self, **kwargs): self.deletions.append(kwargs)

    class Store:
        def __init__(self): self.collection = Collection(); self.added = []
        def add_chunks(self, chunks): self.added.extend(chunks); return len(chunks)

    store = Store()
    report = DailySitemapSyncer(Parser(), Crawler(), vector_store=store,
                                snapshot_path=tmp_path / "snapshot.json").sync()
    assert report.status == "partial"
    assert report.failed_articles == ["1"]
    assert store.collection.deletions == []


def test_daily_scheduler_calculates_configured_local_run_time():
    from datetime import datetime
    from task1_knowledge_brain.syncer.scheduler import DailySyncScheduler
    scheduler = DailySyncScheduler(syncer=SimpleNamespace(), run_time="03:15", timezone_name="Asia/Kolkata")
    now = datetime.fromisoformat("2026-10-18T02:15:00+05:30")
    assert scheduler.next_run(now).isoformat() == "2026-10-18T03:15:00+05:30"


def test_crawl_checkpoint_persists_completion_and_index_state(tmp_path):
    from task1_knowledge_brain.scraper.checkpoint import CrawlCheckpoint
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = CrawlCheckpoint(checkpoint_path)
    url = "https://support.brilliantdirectories.com/support/solutions/articles/123-test"
    checkpoint.mark_completed(url, "123", "2026-10-18")
    checkpoint.mark_indexed(url, "123", "content-digest", 3)
    resumed = CrawlCheckpoint(checkpoint_path)
    assert resumed.is_completed(url)
    assert resumed.data["indexed_articles"][url]["content_hash"] == "content-digest"
    assert resumed.filter_pending([url, "https://support.brilliantdirectories.com/other"]) == [
        "https://support.brilliantdirectories.com/other"
    ]


def test_task1_api_routes_with_isolated_services(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from task1_knowledge_brain.main import app
    from task1_knowledge_brain.api import routes
    from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
    from task1_knowledge_brain.syncer.daily_sync import SyncReport
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    tickets = TicketManager(tmp_path / "api_tickets.json")
    store = BDVectorStore(tmp_path / "api_chroma")
    routes.vector_store = store
    routes.ticket_manager = tickets
    routes.query_engine = BDQueryEngine(retriever=StubRetriever([doc()]), ticket_manager=tickets)
    routes.feedback_ingester = FeedbackIngester(store, tickets)

    class SyncStub:
        def sync(self, max_updates=None):
            return SyncReport("success", "test-time", ["a"], ["b"], ["c"], "ok", 4, [])
    routes.daily_syncer = SyncStub()

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/v1/health").json()["chroma_db_healthy"] is True
        assert client.post("/api/v1/query", json={"query": "   "}).status_code == 422
        assert client.post("/api/v1/query", json={"query": "email accounts", "top_k": 11}).status_code == 422
        assert client.post("/api/v1/query", json={"query": "How do I create and manage email accounts?", "top_k": 2}).json()["has_answer"] is True
        assert client.post("/api/v1/query", json={"query": "How do I manage email accounts?", "category_filter": "Emails"}).json()["has_answer"] is True
        unsupported = client.post("/api/v1/query", json={"query": "How do I configure orbital flight paths?"}).json()
        assert unsupported["has_answer"] is False and unsupported["ticket_id"]
        ticket_id = unsupported["ticket_id"]
        listed = client.get("/api/v1/tickets?status=pending")
        assert listed.status_code == 200 and any(t["ticket_id"] == ticket_id for t in listed.json())
        assert client.get(f"/api/v1/tickets/{ticket_id}").status_code == 200
        resolved = client.post(f"/api/v1/tickets/{ticket_id}/resolve", json={
            "expert_answer": "Use the verified orbital path configuration page to set each flight waypoint.",
            "expert_name": "Verified Reviewer", "category": "Flight Settings"})
        assert resolved.status_code == 200 and resolved.json()["indexed_chunks"] == 1
        settings = client.post("/api/v1/settings/check", json={"feature_description": "email accounts"})
        assert settings.status_code == 200
        sync = client.post("/api/v1/sync", json={"max_updates": 1})
        assert sync.status_code == 200 and sync.json()["added_count"] == 1
        assert client.post("/api/v1/tickets/missing/resolve", json={"expert_answer": "This is a sufficiently long answer."}).status_code == 404
        pending = client.post("/api/v1/query", json={"query": "How do I configure orbital navigation paths?"}).json()
        class FailingIngester:
            def ingest_expert_answer(self, **kwargs): raise RuntimeError("simulated index failure")
        routes.feedback_ingester = FailingIngester()
        failure = client.post(f"/api/v1/tickets/{pending['ticket_id']}/resolve", json={
            "expert_answer": "This expert answer is valid but cannot be stored right now."})
        assert failure.status_code == 500
        assert client.get(f"/api/v1/tickets/{pending['ticket_id']}").json()["status"] == "pending"
