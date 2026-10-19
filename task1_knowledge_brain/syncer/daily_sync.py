"""Conservative, retryable sitemap synchronization for Task 1."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from task1_knowledge_brain.scraper.sitemap_parser import SitemapParser, SitemapEntry
from task1_knowledge_brain.scraper.crawler import PoliteCrawler
from task1_knowledge_brain.knowledge_base.chunker import HierarchyAwareChunker
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    status: str
    timestamp: str
    added_articles: List[str]
    updated_articles: List[str]
    removed_articles: List[str]
    message: str
    unchanged_count: int = 0
    failed_articles: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DailySitemapSyncer:
    """Apply a complete sitemap snapshot; require two complete misses to delete."""
    def __init__(self, sitemap_parser=None, crawler=None, chunker=None, vector_store=None,
                 snapshot_path: Optional[Path] = None):
        self.sitemap_parser = sitemap_parser or SitemapParser()
        self.crawler = crawler or PoliteCrawler()
        self.chunker = chunker or HierarchyAwareChunker(target_chunk_size=600)
        self.vector_store = vector_store or BDVectorStore()
        self.snapshot_path = Path(snapshot_path or DATA_DIR / "sync_snapshot.json")
        self._sync_lock = threading.Lock()

    @staticmethod
    def _hash(article) -> str:
        raw = json.dumps(article.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_snapshot(self) -> Dict[str, Any]:
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"urls": [], "missing": {}, "observed_on": None}

    def _save_snapshot(self, data: Dict[str, Any]) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.snapshot_path)

    def sync(self, max_updates: Optional[int] = None) -> SyncReport:
        if not self._sync_lock.acquire(blocking=False):
            return SyncReport("skipped", datetime.now(timezone.utc).isoformat(), [], [], [],
                              "Another sitemap sync is already running.", 0, [])
        try:
            return self._sync_locked(max_updates)
        finally:
            self._sync_lock.release()

    def _sync_locked(self, max_updates: Optional[int] = None) -> SyncReport:
        stamp = datetime.now(timezone.utc).isoformat()
        def abort(msg, failed=None):
            return SyncReport("aborted", stamp, [], [], [], msg, 0, failed or [])
        try:
            entries = self.sitemap_parser.get_articles_only(self.sitemap_parser.parse())
            if len(entries) < 100:
                return abort(f"Incomplete sitemap snapshot ({len(entries)} article URLs; minimum 100). No changes applied.")
            urls = [e.loc for e in entries]
            ids = [str(e.article_id or "") for e in entries]
            if any(not x for x in ids) or len(urls) != len(set(urls)) or len(ids) != len(set(ids)):
                return abort("Sitemap contains missing or duplicate article URLs/IDs. No changes applied.")
            previous = self._load_snapshot()
            old_urls = set(previous.get("urls", []))
            if old_urls and len(urls) < max(100, int(len(old_urls) * .90)):
                return abort(f"Sitemap shrank from {len(old_urls)} to {len(urls)} URLs; refusing a potentially partial snapshot.")
        except Exception as exc:
            logger.exception("Sitemap snapshot failed")
            return abort(f"Sitemap fetch/parse failed ({type(exc).__name__}: {exc}); no changes applied.")

        remote = {str(e.article_id): e for e in entries}
        local = {str(a.article_id): a for a in self.crawler.load_all_saved_articles()}
        additions = [key for key in remote if key not in local]
        updates = [key for key in remote if key in local and remote[key].lastmod != local[key].lastmod
                   and (remote[key].lastmod is not None or local[key].lastmod is not None)]
        unchanged = len(remote) - len(additions) - len(updates)
        work = additions + updates
        if max_updates is not None:
            work = work[:max_updates]
        succeeded_add, succeeded_update, failed = [], [], []
        for article_id in work:
            entry = remote[article_id]
            try:
                article = self.crawler.crawl_article(entry)
                if not article:
                    raise RuntimeError("crawler returned no article")
                chunks = self.chunker.chunk_article(article)
                if not chunks:
                    raise ValueError("replacement article yielded no indexable chunks")
                # Commit replacement first; preserve old vectors on any crawl/chunk failure.
                self.vector_store.add_chunks(chunks)
                get_ids = getattr(self.vector_store, "get_chunk_ids_for_article", None)
                if callable(get_ids):
                    current_ids = get_ids(article_id)
                else:  # Keep compatibility with older test doubles/adapters.
                    current = self.vector_store.collection.get(where={"article_id": article_id}, include=[])
                    current_ids = current.get("ids", [])
                fresh = {c.chunk_id for c in chunks}
                stale = [cid for cid in current_ids if cid not in fresh]
                if stale:
                    delete_ids = getattr(self.vector_store, "delete_chunk_ids", None)
                    if callable(delete_ids):
                        delete_ids(stale)
                    else:
                        self.vector_store.collection.delete(ids=stale)
                (succeeded_add if article_id in additions else succeeded_update).append(article_id)
            except Exception as exc:
                failed.append(article_id)
                logger.exception("Sync failed for article %s", article_id)

        incomplete = len(work) < len(additions) + len(updates) or bool(failed)
        if incomplete:
            return SyncReport("partial", stamp, succeeded_add, succeeded_update, [],
                              "Snapshot changes were processed partially; deletion snapshot was not advanced.",
                              unchanged, failed)

        today = datetime.now().astimezone().date().isoformat()
        old_ids = {str(Path(u).name) for u in previous.get("urls", [])}
        old_ids = {x.split("-", 1)[0] for x in old_ids}
        present_ids = set(remote)
        missing = dict(previous.get("missing", {}))
        # Count at most one complete observation per local day.
        if previous.get("observed_on") != today:
            for article_id in (old_ids | set(missing)) - present_ids:
                missing[article_id] = int(missing.get(article_id, 0)) + 1
            for article_id in present_ids:
                missing.pop(article_id, None)
        removed = []
        for article_id, misses in list(missing.items()):
            if misses >= 2:
                # Collection deletion uses metadata filter; preserve source files if DB deletion fails.
                delete_article = getattr(self.vector_store, "delete_article", None)
                if callable(delete_article):
                    delete_article(article_id)
                else:
                    self.vector_store.collection.delete(where={"article_id": article_id})
                file_path = self.crawler.articles_dir / f"{article_id}.json"
                if file_path.exists():
                    file_path.unlink()
                removed.append(article_id)
                del missing[article_id]
        self._save_snapshot({"urls": urls, "missing": missing, "observed_on": today, "updated_at": stamp})
        return SyncReport("success", stamp, succeeded_add, succeeded_update, removed,
                          f"Sync complete: added {len(succeeded_add)}, updated {len(succeeded_update)}, unchanged {unchanged}, removed {len(removed)}, failed 0.",
                          unchanged, [])
