"""
Checkpoint Manager for Polite Crawler.
Maintains persistent state so scraping can safely stop halfway and resume
without re-downloading already processed articles.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Set, Any

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "data" / "crawler_checkpoint.json"


class CrawlCheckpoint:
    def __init__(self, filepath: Path = DEFAULT_CHECKPOINT_PATH):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, Any] = {
            "started_at": None,
            "last_updated_at": None,
            "completed_articles": {},  # url -> {article_id, scraped_at, lastmod}
            "failed_articles": {},     # url -> {error, retries, last_attempt}
            "indexed_articles": {},    # url -> {article_id, content_hash, indexed_at, chunks}
            "index_failed_articles": {},
        }
        self.load()

    def load(self) -> None:
        """Loads state from JSON checkpoint file if it exists."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                logger.info(
                    f"Loaded checkpoint with {len(self.data.get('completed_articles', {}))} completed articles."
                )
            except Exception as e:
                logger.error(f"Failed to read checkpoint at {self.filepath}: {e}. Initializing fresh state.")

    def save(self) -> None:
        """Atomically saves current crawler state to JSON."""
        self.data["last_updated_at"] = datetime.utcnow().isoformat()
        temp_path = self.filepath.with_name(
            f"{self.filepath.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        for attempt in range(4):
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                temp_path.replace(self.filepath)
                return
            except PermissionError:
                if attempt == 3:
                    logger.exception("Failed to replace checkpoint after bounded lock retries")
                    raise
                time.sleep(0.05 * (2 ** attempt))
            except Exception as e:
                logger.error("Failed to save checkpoint (%s)", type(e).__name__)
                raise

    def is_completed(self, url: str) -> bool:
        """Checks if a URL has already been successfully crawled."""
        return url in self.data.get("completed_articles", {})

    def mark_completed(
        self,
        url: str,
        article_id: Optional[str] = None,
        lastmod: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """Records an article as successfully processed."""
        if self.data["started_at"] is None:
            self.data["started_at"] = datetime.utcnow().isoformat()

        # Remove from failed if it was there
        self.data.get("failed_articles", {}).pop(url, None)

        self.data.setdefault("completed_articles", {})[url] = {
            "article_id": article_id,
            "lastmod": lastmod,
            "scraped_at": datetime.utcnow().isoformat(),
            **(extra or {}),
        }
        self.save()

    def mark_failed(self, url: str, error: str) -> None:
        """Records an article download/parse failure with retry count."""
        failed = self.data.setdefault("failed_articles", {})
        prior = failed.get(url, {"retries": 0})
        prior["error"] = error
        prior["retries"] = prior.get("retries", 0) + 1
        prior["last_attempt"] = datetime.utcnow().isoformat()
        failed[url] = prior
        self.save()

    def mark_indexed(self, url: str, article_id: str, content_hash: str, chunks: int) -> None:
        self.data.setdefault("index_failed_articles", {}).pop(url, None)
        self.data.setdefault("indexed_articles", {})[url] = {
            "article_id": str(article_id), "content_hash": content_hash,
            "indexed_at": datetime.utcnow().isoformat(), "chunks": int(chunks),
        }
        self.save()

    def mark_index_failed(self, url: str, error: str) -> None:
        failures = self.data.setdefault("index_failed_articles", {})
        previous = failures.get(url, {"retries": 0})
        previous.update({"error": error, "retries": previous.get("retries", 0) + 1,
                         "last_attempt": datetime.utcnow().isoformat()})
        failures[url] = previous
        self.save()

    def filter_pending(self, urls: List[str]) -> List[str]:
        """Returns only the URLs that have not yet been marked completed."""
        completed = set(self.data.get("completed_articles", {}).keys())
        return [u for u in urls if u not in completed]

    def get_progress(self, total: Optional[int] = None) -> Dict[str, Any]:
        """Returns statistics on crawling progress."""
        completed_count = len(self.data.get("completed_articles", {}))
        failed_count = len(self.data.get("failed_articles", {}))
        progress: Dict[str, Any] = {
            "completed": completed_count,
            "failed": failed_count,
            "last_updated_at": self.data.get("last_updated_at"),
            "started_at": self.data.get("started_at"),
        }
        if total is not None and total > 0:
            progress["total"] = total
            progress["remaining"] = max(0, total - completed_count)
            progress["percent"] = round((completed_count / total) * 100, 2)
        return progress

    def reset(self) -> None:
        """Resets the checkpoint state completely."""
        self.data = {
            "started_at": None,
            "last_updated_at": None,
            "completed_articles": {},
            "failed_articles": {},
            "indexed_articles": {},
            "index_failed_articles": {},
        }
        self.save()
