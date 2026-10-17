"""
Sitemap Parser for Brilliant Directories Solution Center.
Fetches and parses https://support.brilliantdirectories.com/support/sitemap.xml
Extracts article URLs, categories, folders, IDs, and last modified dates.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import re
import xml.etree.ElementTree as ET
import httpx


SITEMAP_URL = "https://support.brilliantdirectories.com/support/sitemap.xml"
DEFAULT_USER_AGENT = (
    "AITechConsultantBot/1.0 (+https://github.com/ai-tech-consultant; research-polite-crawler)"
)


@dataclass
class SitemapEntry:
    loc: str
    lastmod: Optional[str] = None
    changefreq: Optional[str] = None
    priority: Optional[float] = None
    is_article: bool = False
    is_category_or_folder: bool = False
    article_id: Optional[str] = None
    slug: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class SitemapParser:
    def __init__(self, sitemap_url: str = SITEMAP_URL, user_agent: str = DEFAULT_USER_AGENT):
        self.sitemap_url = sitemap_url
        self.user_agent = user_agent

    def fetch_xml(self) -> str:
        """Fetches raw sitemap XML content politely with a custom user agent."""
        headers = {"User-Agent": self.user_agent}
        with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
            response = client.get(self.sitemap_url)
            response.raise_for_status()
            return response.text

    def parse(self, xml_content: Optional[str] = None) -> List[SitemapEntry]:
        """
        Parses sitemap XML string (or fetches if None) into structured SitemapEntry objects.
        """
        if xml_content is None:
            xml_content = self.fetch_xml()

        root = ET.fromstring(xml_content)
        # Sitemaps use XML namespace
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        entries: List[SitemapEntry] = []

        for url_node in root.findall("ns:url", namespace):
            loc_node = url_node.find("ns:loc", namespace)
            if loc_node is None or not loc_node.text:
                continue

            loc = loc_node.text.strip()
            lastmod_node = url_node.find("ns:lastmod", namespace)
            changefreq_node = url_node.find("ns:changefreq", namespace)
            priority_node = url_node.find("ns:priority", namespace)

            lastmod = lastmod_node.text.strip() if lastmod_node is not None and lastmod_node.text else None
            changefreq = changefreq_node.text.strip() if changefreq_node is not None and changefreq_node.text else None
            priority = float(priority_node.text.strip()) if priority_node is not None and priority_node.text else None

            # Pattern: /support/solutions/articles/{article_id}-{slug}
            article_match = re.search(r"/support/solutions/articles/(\d+)(?:-(.*))?$", loc)
            is_article = bool(article_match)
            article_id = article_match.group(1) if article_match else None
            slug = article_match.group(2) if article_match else None

            # Pattern: /support/solutions/{id} (Category or Folder list)
            cat_match = re.search(r"/support/solutions/(\d+)$", loc)
            is_category_or_folder = bool(cat_match)

            entries.append(
                SitemapEntry(
                    loc=loc,
                    lastmod=lastmod,
                    changefreq=changefreq,
                    priority=priority,
                    is_article=is_article,
                    is_category_or_folder=is_category_or_folder,
                    article_id=article_id,
                    slug=slug,
                )
            )

        return entries

    def get_articles_only(self, entries: Optional[List[SitemapEntry]] = None) -> List[SitemapEntry]:
        """Filters entries to return only support articles."""
        if entries is None:
            entries = self.parse()
        return [entry for entry in entries if entry.is_article]

    def get_categories_and_folders(self, entries: Optional[List[SitemapEntry]] = None) -> List[SitemapEntry]:
        """Filters entries to return categories and folder index pages."""
        if entries is None:
            entries = self.parse()
        return [entry for entry in entries if entry.is_category_or_folder]
