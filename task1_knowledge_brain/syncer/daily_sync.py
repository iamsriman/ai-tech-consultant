"""
Daily Sitemap Differential Synchronization for Task 1: Knowledge Brain.
Detects added, updated, and removed articles in the Brilliant Directories Solution Center.
Guarantees safety: NEVER deletes articles if sitemap fetching or parsing fails.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from task1_knowledge_brain.scraper.sitemap_parser import SitemapParser, SitemapEntry
from task1_knowledge_brain.scraper.crawler import PoliteCrawler
from task1_knowledge_brain.knowledge_base.chunker import HierarchyAwareChunker
from task1_knowledge_brain.knowledge_base.vector_store import BDVectorStore

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    status: str  # "success", "aborted", "skipped"
    timestamp: str
    added_articles: List[str]
    updated_articles: List[str]
    removed_articles: List[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DailySitemapSyncer:
    """
    Synchronizes local article cache and ChromaDB vector store with the live sitemap.
    """

    def __init__(
        self,
        sitemap_parser: Optional[SitemapParser] = None,
        crawler: Optional[PoliteCrawler] = None,
        chunker: Optional[HierarchyAwareChunker] = None,
        vector_store: Optional[BDVectorStore] = None,
    ):
        self.sitemap_parser = sitemap_parser or SitemapParser()
        self.crawler = crawler or PoliteCrawler()
        self.chunker = chunker or HierarchyAwareChunker(target_chunk_size=600)
        self.vector_store = vector_store or BDVectorStore()

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Computes a SHA-256 hash of article text to detect meaningful changes."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sync(self, max_updates: Optional[int] = None) -> SyncReport:
        """
        Executes a safe daily differential sync.
        CRITICAL RULE: If sitemap fetching or parsing encounters ANY error,
        the sync aborts immediately without touching or deleting any existing articles.
        """
        timestamp = datetime.utcnow().isoformat()

        # Step 1: Safe fetch and parse of sitemap
        try:
            logger.info("Daily Sync: Fetching and parsing live sitemap...")
            entries = self.sitemap_parser.parse()
            remote_articles = self.sitemap_parser.get_articles_only(entries)

            if not remote_articles or len(remote_articles) < 100:
                # Sanity check: Help center has >1,000 articles. If fewer than 100 parsed, abort!
                msg = f"Sitemap parsing yielded suspiciously low article count ({len(remote_articles)}). Aborting sync for data safety."
                logger.error(msg)
                return SyncReport(
                    status="aborted",
                    timestamp=timestamp,
                    added_articles=[],
                    updated_articles=[],
                    removed_articles=[],
                    message=msg,
                )

        except Exception as e:
            msg = f"Sitemap fetch/parse failed: {e}. Aborting sync safely without deleting any data."
            logger.critical(msg)
            return SyncReport(
                status="aborted",
                timestamp=timestamp,
                added_articles=[],
                updated_articles=[],
                removed_articles=[],
                message=msg,
            )

        # Step 2: Index remote articles by ID
        remote_map: Dict[str, SitemapEntry] = {}
        for entry in remote_articles:
            if entry.article_id:
                remote_map[entry.article_id] = entry

        # Step 3: Load existing local inventory
        local_articles_list = self.crawler.load_all_saved_articles()
        local_map = {art.article_id: art for art in local_articles_list}
        logger.info(
            f"Daily Sync: {len(remote_map)} articles in sitemap, {len(local_map)} local articles currently saved."
        )

        added: List[str] = []
        updated: List[str] = []
        removed: List[str] = []

        # Step 4: Detect Added and Updated
        for art_id, remote_entry in remote_map.items():
            if art_id not in local_map:
                added.append(art_id)
            else:
                local_art = local_map[art_id]
                # Compare lastmod
                if remote_entry.lastmod and local_art.lastmod and remote_entry.lastmod != local_art.lastmod:
                    updated.append(art_id)

        # Step 5: Detect Removed (Local exists, but no longer present in remote sitemap)
        for art_id in local_map:
            if art_id not in remote_map:
                removed.append(art_id)

        logger.info(
            f"Daily Sync Diff: {len(added)} new, {len(updated)} updated, {len(removed)} removed."
        )

        # Step 6: Process Removed articles (Safely remove from ChromaDB and disk)
        for art_id in removed:
            self.vector_store.delete_article(art_id)
            art_file = self.crawler.articles_dir / f"{art_id}.json"
            if art_file.exists():
                art_file.unlink()
            logger.info(f"Removed article {art_id} from ChromaDB and local disk.")

        # Step 7: Process Updated articles (Delete old chunks first, then re-index)
        processed_updates = updated[:max_updates] if max_updates else updated
        for art_id in processed_updates:
            entry = remote_map[art_id]
            # 1. Delete old chunks from ChromaDB
            self.vector_store.delete_article(art_id)
            # 2. Re-crawl
            new_art = self.crawler.crawl_article(entry)
            if new_art:
                # 3. Re-chunk and re-index
                new_chunks = self.chunker.chunk_article(new_art)
                self.vector_store.add_chunks(new_chunks)
                logger.info(f"Updated article {art_id} with {len(new_chunks)} fresh chunks.")

        # Step 8: Process New articles
        processed_added = added[:max_updates] if max_updates else added
        for art_id in processed_added:
            entry = remote_map[art_id]
            new_art = self.crawler.crawl_article(entry)
            if new_art:
                new_chunks = self.chunker.chunk_article(new_art)
                self.vector_store.add_chunks(new_chunks)
                logger.info(f"Added new article {art_id} with {len(new_chunks)} chunks.")

        report = SyncReport(
            status="success",
            timestamp=timestamp,
            added_articles=processed_added,
            updated_articles=processed_updates,
            removed_articles=removed,
            message=(
                f"Sync complete. Added {len(processed_added)}, "
                f"updated {len(processed_updates)}, removed {len(removed)}."
            ),
        )
        return report
