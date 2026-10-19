"""Polite, resumable sitemap crawl with incremental Chroma indexing."""
import argparse
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

from task1_knowledge_brain.knowledge_base.chunker import HierarchyAwareChunker
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore
from task1_knowledge_brain.scraper.crawler import PoliteCrawler
from task1_knowledge_brain.scraper.sitemap_parser import SitemapEntry

logger = logging.getLogger("knowledge_brain.crawl")


def article_hash(article) -> str:
    payload = json.dumps(article.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_article(article, store: BDVectorStore, chunker: HierarchyAwareChunker) -> int:
    chunks = chunker.chunk_article(article)
    if not chunks:
        raise ValueError("Article produced no indexable chunks")
    # Upsert replacements first; only then remove obsolete chunk IDs.
    store.add_chunks(chunks)
    get_ids = getattr(store, "get_chunk_ids_for_article", None)
    if callable(get_ids):
        current_ids = get_ids(str(article.article_id))
    else:
        current = store.collection.get(where={"article_id": str(article.article_id)}, include=[])
        current_ids = current.get("ids", [])
    new_ids = {chunk.chunk_id for chunk in chunks}
    stale_ids = [chunk_id for chunk_id in current_ids if chunk_id not in new_ids]
    if stale_ids:
        delete_ids = getattr(store, "delete_chunk_ids", None)
        if callable(delete_ids):
            delete_ids(stale_ids)
        else:
            store.collection.delete(ids=stale_ids)
    return len(chunks)


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new URLs processed this run (default: all pending)")
    parser.add_argument("--batch-size", type=int, default=10, help="Articles per progress/index batch")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    crawler = PoliteCrawler()
    chunker = HierarchyAwareChunker()
    store = BDVectorStore()
    entries = crawler.sitemap_parser.get_articles_only(crawler.sitemap_parser.parse())
    if len(entries) < 100:
        raise RuntimeError(f"Sitemap yielded only {len(entries)} article URLs; refusing an incomplete crawl snapshot")
    # Sitemap duplicates or conflicting article IDs make deletion/indexing unsafe.
    urls = [entry.loc for entry in entries]
    ids = [entry.article_id for entry in entries]
    if len(urls) != len(set(urls)) or len(ids) != len(set(ids)):
        raise RuntimeError("Sitemap contains duplicate article URLs or IDs; refusing to crawl/index ambiguous records")

    by_url = {entry.loc: entry for entry in entries}
    saved = crawler.load_all_saved_articles()
    saved_by_url = {article.url: article for article in saved}
    indexed = crawler.checkpoint.data.setdefault("indexed_articles", {})
    index_failed = crawler.checkpoint.data.setdefault("index_failed_articles", {})
    indexed_this_run = 0
    # Reconcile saved articles first, including downloads saved just before a crash.
    for article in saved:
        url = article.url
        digest = article_hash(article)
        record = indexed.get(url, {})
        if record.get("content_hash") == digest:
            # A prior checkpoint write can fail after the Chroma upsert. Clear
            # its failure markers once the saved digest proves the index is current.
            if url in index_failed:
                crawler.checkpoint.mark_indexed(url, article.article_id, digest, record.get("chunks", 0))
            if (not crawler.checkpoint.is_completed(url)
                    or url in crawler.checkpoint.data.get("failed_articles", {})):
                crawler.checkpoint.mark_completed(url, article.article_id, article.lastmod,
                                                  {"title": article.title, "reconciled_from_disk": True})
            continue
        try:
            chunk_count = index_article(article, store, chunker)
            crawler.checkpoint.mark_indexed(url, article.article_id, digest, chunk_count)
            indexed_this_run += chunk_count
            if url in by_url and not crawler.checkpoint.is_completed(url):
                crawler.checkpoint.mark_completed(url, article.article_id, article.lastmod,
                                                  {"title": article.title, "reconciled_from_disk": True})
        except Exception as exc:
            crawler.checkpoint.mark_index_failed(url, f"{type(exc).__name__}: {exc}")
            logger.exception("Could not index saved article %s", article.article_id)

    pending = []
    for entry in entries:
        article = saved_by_url.get(entry.loc)
        indexed_record = indexed.get(entry.loc, {})
        saved_is_current = bool(article and (
            not entry.lastmod or not article.lastmod or entry.lastmod == article.lastmod
        ))
        is_indexed = bool(article and indexed_record.get("content_hash") == article_hash(article))
        if not (crawler.checkpoint.is_completed(entry.loc) and saved_is_current and is_indexed):
            pending.append(entry)
    if args.limit is not None:
        pending = pending[:args.limit]
    skipped = len(entries) - len(pending)
    newly_crawled = 0
    failed_this_run = 0
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset + args.batch_size]
        for entry in batch:
            try:
                article = saved_by_url.get(entry.loc)
                if article is None or (entry.lastmod and article.lastmod and entry.lastmod != article.lastmod):
                    article = crawler.crawl_article(entry)
                    if article is None:
                        failed_this_run += 1
                        continue
                    newly_crawled += 1
                digest = article_hash(article)
                chunk_count = index_article(article, store, chunker)
                crawler.checkpoint.mark_indexed(entry.loc, article.article_id, digest, chunk_count)
                indexed_this_run += chunk_count
                if not crawler.checkpoint.is_completed(entry.loc):
                    crawler.checkpoint.mark_completed(entry.loc, article.article_id, entry.lastmod,
                                                      {"title": article.title})
            except Exception as exc:
                crawler.checkpoint.mark_index_failed(entry.loc, f"{type(exc).__name__}: {exc}")
                failed_this_run += 1
                logger.exception("Crawl/index failed for %s", entry.loc)
        report = {
            "discovered": len(entries), "completed_checkpoint": len(crawler.checkpoint.data.get("completed_articles", {})),
            "skipped_completed": skipped, "crawled_this_run": newly_crawled,
            "failed_this_run": failed_this_run, "chunks_indexed_this_run": indexed_this_run,
            "indexed_url_count": len(crawler.checkpoint.data.get("indexed_articles", {})),
            "index_failed_url_count": len(crawler.checkpoint.data.get("index_failed_articles", {})),
            "remaining": max(0, len(entries) - len(crawler.checkpoint.data.get("completed_articles", {}))),
            "indexed_chunks_total": store.count(), "batch": f"{min(offset + len(batch), len(pending))}/{len(pending)}",
        }
        print(json.dumps(report), flush=True)
    if not pending:
        print(json.dumps({
            "discovered": len(entries), "completed_checkpoint": len(crawler.checkpoint.data.get("completed_articles", {})),
            "skipped_completed": skipped, "crawled_this_run": 0, "failed_this_run": 0,
            "chunks_indexed_this_run": indexed_this_run,
            "indexed_url_count": len(crawler.checkpoint.data.get("indexed_articles", {})),
            "index_failed_url_count": len(crawler.checkpoint.data.get("index_failed_articles", {})),
            "remaining": max(0, len(entries) - len(crawler.checkpoint.data.get("completed_articles", {}))),
            "indexed_chunks_total": store.count(),
        }), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
