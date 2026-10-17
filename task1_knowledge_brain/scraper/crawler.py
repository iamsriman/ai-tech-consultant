"""
Polite Web Crawler for Brilliant Directories Solution Center.
Crawls solution articles with rate limiting, checkpointing, and structured extraction.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional
import httpx

from .sitemap_parser import SitemapParser, SitemapEntry
from .rate_limiter import PoliteRateLimiter
from .checkpoint import CrawlCheckpoint
from task1_knowledge_brain.multimodal.image_extractor import ArticleData, ArticleParser, ExtractedImage

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("task1_knowledge_brain/data")
ARTICLES_DIR = DEFAULT_DATA_DIR / "articles"


class PoliteCrawler:
    """
    Polite and resumable crawler for support.brilliantdirectories.com
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        rate_limiter: Optional[PoliteRateLimiter] = None,
        checkpoint: Optional[CrawlCheckpoint] = None,
        parser: Optional[ArticleParser] = None,
    ):
        self.data_dir = Path(data_dir)
        self.articles_dir = self.data_dir / "articles"
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limiter = rate_limiter or PoliteRateLimiter(min_delay=0.4, max_delay=0.8)
        self.checkpoint = checkpoint or CrawlCheckpoint(self.data_dir / "crawler_checkpoint.json")
        self.article_parser = parser or ArticleParser()
        self.sitemap_parser = SitemapParser()

    def fetch_page_html(self, url: str) -> str:
        """Fetches raw page HTML with rate limiting and retry handling."""
        def _fetch() -> str:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            with httpx.Client(timeout=25.0, follow_redirects=True, headers=headers) as client:
                res = client.get(url)
                res.raise_for_status()
                return res.text

        return self.rate_limiter.execute_with_retry(_fetch)

    def crawl_article(self, entry: SitemapEntry) -> Optional[ArticleData]:
        """Crawls and parses a single article, saving it to disk."""
        url = entry.loc
        article_id = entry.article_id or url.split("/")[-1]

        try:
            html = self.fetch_page_html(url)
            article_data = self.article_parser.parse(
                html=html,
                url=url,
                article_id=article_id,
                lastmod=entry.lastmod,
                scraped_at=datetime.utcnow().isoformat(),
            )

            # Save structured article JSON
            article_file = self.articles_dir / f"{article_id}.json"
            with open(article_file, "w", encoding="utf-8") as f:
                json.dump(article_data.to_dict(), f, indent=2, ensure_ascii=False)

            # Update checkpoint
            self.checkpoint.mark_completed(
                url=url,
                article_id=article_id,
                lastmod=entry.lastmod,
                extra={"title": article_data.title, "images_count": len(article_data.images)},
            )
            return article_data

        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            self.checkpoint.mark_failed(url=url, error=str(e))
            return None

    def crawl_all(
        self,
        limit: Optional[int] = None,
        batch_size: int = 10,
        resume: bool = True,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> List[ArticleData]:
        """
        Crawls articles from the sitemap.
        - limit: max articles to crawl (useful for testing or batch runs)
        - resume: if True, skips articles already recorded as completed
        """
        all_entries = self.sitemap_parser.parse()
        article_entries = self.sitemap_parser.get_articles_only(all_entries)

        if resume:
            # Filter out entries whose URLs are already completed
            pending_entries = [e for e in article_entries if not self.checkpoint.is_completed(e.loc)]
        else:
            self.checkpoint.reset()
            pending_entries = article_entries

        if limit is not None:
            pending_entries = pending_entries[:limit]

        logger.info(
            f"Starting crawl: {len(pending_entries)} articles pending out of {len(article_entries)} total."
        )

        results: List[ArticleData] = []
        for idx, entry in enumerate(pending_entries, 1):
            article = self.crawl_article(entry)
            if article:
                results.append(article)

            if on_progress:
                stats = self.checkpoint.get_progress(total=len(article_entries))
                stats["current_batch_step"] = f"{idx}/{len(pending_entries)}"
                on_progress(stats)

        return results

    def load_article_from_disk(self, article_id: str) -> Optional[ArticleData]:
        """Loads an already-saved article JSON from disk."""
        article_file = self.articles_dir / f"{article_id}.json"
        if not article_file.exists():
            return None
        with open(article_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = [
            ExtractedImage(**img) if isinstance(img, dict) else img
            for img in data.get("images", [])
        ]
        return ArticleData(
            article_id=data["article_id"],
            url=data["url"],
            title=data["title"],
            category=data["category"],
            folder=data["folder"],
            breadcrumbs=data["breadcrumbs"],
            content=data["content"],
            images=images,
            links=data["links"],
            lastmod=data.get("lastmod"),
            scraped_at=data.get("scraped_at"),
        )

    def load_all_saved_articles(self) -> List[ArticleData]:
        """Loads all locally saved articles from the data directory."""
        articles: List[ArticleData] = []
        for file in self.articles_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                images = [
                    ExtractedImage(**img) if isinstance(img, dict) else img
                    for img in data.get("images", [])
                ]
                articles.append(
                    ArticleData(
                        article_id=data["article_id"],
                        url=data["url"],
                        title=data["title"],
                        category=data["category"],
                        folder=data["folder"],
                        breadcrumbs=data["breadcrumbs"],
                        content=data["content"],
                        images=images,
                        links=data["links"],
                        lastmod=data.get("lastmod"),
                        scraped_at=data.get("scraped_at"),
                    )
                )
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        return articles
