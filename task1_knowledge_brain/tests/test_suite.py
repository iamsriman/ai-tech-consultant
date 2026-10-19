# Test suite for Task 1 Knowledge Brain

import pytest
import os
from pathlib import Path

# Ensure project root is on sys.path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from task1_knowledge_brain.knowledge_base.chunker import HierarchyAwareChunker
from task1_knowledge_brain.scraper.crawler import PoliteCrawler
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
from task1_knowledge_brain.knowledge_base.retriever import KnowledgeRetriever
from task1_knowledge_brain.knowledge_base.query_engine import BDQueryEngine
from task1_knowledge_brain.syncer.daily_sync import DailySitemapSyncer
from task1_knowledge_brain.scraper.sitemap_parser import SitemapEntry
from task1_knowledge_brain.human_loop.ticket_manager import TicketManager

@pytest.fixture(scope="module")
def crawler():
    """PoliteCrawler with default configuration."""
    return PoliteCrawler()

@pytest.fixture(scope="module")
def articles(crawler):
    """Load all saved articles. The test suite expects at least one article already
    scraped in the workspace (previous runs of the crawler)."""
    arts = crawler.load_all_saved_articles()
    assert len(arts) > 0, "No articles found on disk – run the crawler before tests."
    return arts

@pytest.fixture(scope="module")
def chunker():
    return HierarchyAwareChunker(target_chunk_size=600, overlap=60)

@pytest.fixture(scope="module")
def all_chunks(chunker, articles):
    chunks = []
    for art in articles:
        chunks.extend(chunker.chunk_article(art))
    # sanity check – at least one chunk per article
    assert len(chunks) >= len(articles)
    return chunks

@pytest.fixture(scope="module")
def vector_store(all_chunks, tmp_path_factory):
    store = BDVectorStore(tmp_path_factory.mktemp("kb-store") / "chroma")
    # Keep this isolated corpus test practical on CPU-only development machines.
    added = store.add_chunks(all_chunks, batch_size=64)
    assert added == len(all_chunks), "All chunks should be upserted"
    return store

def test_chunker_injects_header(all_chunks):
    """Each chunk must begin with the hierarchy header and contain URL info."""
    sample = all_chunks[0]
    assert sample.text.startswith("[Article:"), "Chunk header missing"
    assert sample.url in sample.text, "Chunk must embed source URL"

def test_vector_store_counts(vector_store, all_chunks):
    """Vector store should report the exact number of stored chunks."""
    assert vector_store.count() == len(all_chunks)

def test_retriever_basic(vector_store):
    retriever = KnowledgeRetriever(vector_store)
    query = "How do I configure email accounts?"
    results = retriever.retrieve(query, top_k=3, min_score=0.2)
    assert isinstance(results, list)
    assert len(results) > 0, "Retriever should return at least one result for a common query"
    top = results[0]
    assert top.score >= 0.0
    assert top.title
    assert top.url.startswith("http")

def test_query_engine_answer(vector_store, tmp_path):
    engine = BDQueryEngine(vector_store=vector_store, ticket_manager=TicketManager(tmp_path / "tickets.json"))
    answer = engine.query("How do I create an email account?", top_k=4)
    assert hasattr(answer, "answer")
    assert hasattr(answer, "has_answer")
    if not answer.has_answer:
        assert answer.ticket_id is not None
    else:
        assert len(answer.citations) > 0

def test_daily_sync_runs_safely_with_mocked_sitemap(tmp_path):
    class Parser:
        def parse(self):
            return [SitemapEntry(loc=f"https://support.brilliantdirectories.com/support/solutions/articles/{i}-x", lastmod=None,
                                 changefreq=None, priority=None, is_article=True, is_category_or_folder=False,
                                 article_id=str(i), slug="x") for i in range(100)]
        @staticmethod
        def get_articles_only(entries):
            return entries
    class Crawler:
        articles_dir = tmp_path / "articles"
        def load_all_saved_articles(self): return []
        def crawl_article(self, entry): return None
    syncer = DailySitemapSyncer(sitemap_parser=Parser(), crawler=Crawler(), vector_store=BDVectorStore(tmp_path / "sync_chroma"))
    report = syncer.sync(max_updates=1)
    assert report.status == "partial", f"Failed article should keep snapshot uncommitted: {report.message}"
    assert isinstance(report.added_articles, list)
    assert isinstance(report.updated_articles, list)
    assert isinstance(report.removed_articles, list)
    assert report.failed_articles == ["0"]
