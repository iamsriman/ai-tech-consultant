"""
Article and Image Content Extractor for Brilliant Directories Support Articles.
Extracts title, hierarchy (Category/Folder), clean structured text, links,
and screenshot images with alt text and surrounding context.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag


@dataclass
class ExtractedImage:
    src: str
    alt: Optional[str] = None
    caption: Optional[str] = None
    is_screenshot: bool = False
    vision_description: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArticleData:
    article_id: str
    url: str
    title: str
    category: str
    folder: str
    breadcrumbs: List[str]
    content: str
    images: List[ExtractedImage] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    lastmod: Optional[str] = None
    scraped_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "article_id": self.article_id,
            "url": self.url,
            "title": self.title,
            "category": self.category,
            "folder": self.folder,
            "breadcrumbs": self.breadcrumbs,
            "content": self.content,
            "images": [img.to_dict() for img in self.images],
            "links": self.links,
            "lastmod": self.lastmod,
            "scraped_at": self.scraped_at,
        }


class ArticleParser:
    """
    Parses raw HTML from Brilliant Directories solution articles into structured ArticleData.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalizes unicode characters, replaces non-breaking spaces and smart quotes/dashes."""
        if not text:
            return ""
        text = text.replace("\u2013", "-").replace("\u2014", "--")
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\xa0", " ")
        return text.strip()

    @staticmethod
    def clean_category(cat: str) -> str:
        """Removes prefixes like 'Section: ' from category names."""
        cat = re.sub(r"^Section:\s*", "", cat, flags=re.IGNORECASE)
        cat = ArticleParser.clean_text(cat)
        return cat.strip()

    def parse(
        self,
        html: str,
        url: str,
        article_id: str,
        lastmod: Optional[str] = None,
        scraped_at: Optional[str] = None,
    ) -> ArticleData:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Title extraction
        title = ""
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.split("|")[0].strip()
        else:
            title = f"Article {article_id}"
        title = self.clean_text(title)

        # 2. Breadcrumbs & Category/Folder extraction
        breadcrumbs: List[str] = []
        breadcrumb_container = soup.select_one(".breadcrumb, ol.breadcrumb, ul.breadcrumb")
        if breadcrumb_container:
            for item in breadcrumb_container.find_all(["li", "a"]):
                text = item.get_text(strip=True)
                if text and text not in breadcrumbs:
                    breadcrumbs.append(text)

        # Structure usually: ['Solution home', 'Category', 'Folder', 'Article Title']
        category = "General"
        folder = "General"

        clean_crumbs = [c for c in breadcrumbs if c.lower() != "solution home"]
        if len(clean_crumbs) >= 3:
            category = self.clean_category(clean_crumbs[0])
            folder = clean_crumbs[1]
        elif len(clean_crumbs) == 2:
            category = self.clean_category(clean_crumbs[0])
            folder = clean_crumbs[0]
        elif len(clean_crumbs) == 1:
            category = self.clean_category(clean_crumbs[0])

        # 3. Article body container
        body_node = soup.select_one(".article-body, .article__body, article, .solution-article")
        if not body_node:
            body_node = soup.select_one(".content") or soup.body or soup

        # 4. Images extraction
        images: List[ExtractedImage] = []
        if body_node:
            for img in body_node.find_all("img"):
                src = img.get("src", "").strip()
                if not src:
                    continue
                # Make relative URL absolute
                full_src = urljoin(url, src)

                # Ignore small tracking icons, spacers, or standard logos
                if any(x in full_src.lower() for x in ["logo", "spacer", "icon", "banner", "1x1", "facebook", "twitter"]):
                    continue

                alt = img.get("alt", "").strip() or None

                # Find surrounding context/caption: nearest paragraph, figcaption, or parent text
                caption = None
                parent = img.parent
                if parent:
                    if parent.name in ["figure", "p", "div"]:
                        caption_text = parent.get_text(strip=True)
                        if caption_text and caption_text != (alt or ""):
                            caption = caption_text[:300]
                    # Also check preceding sibling heading or text
                    prev = img.find_previous(["h2", "h3", "h4", "p"])
                    if prev and not caption:
                        prev_text = prev.get_text(strip=True)
                        if prev_text:
                            caption = f"Context: {prev_text[:200]}"

                is_screenshot = bool(
                    "freshdesk.com" in full_src
                    or "attachment" in full_src
                    or "uploads" in full_src
                    or (alt and len(alt) > 15)
                )

                images.append(
                    ExtractedImage(
                        src=full_src,
                        alt=alt,
                        caption=caption,
                        is_screenshot=is_screenshot,
                    )
                )

        # 5. Links extraction
        links: List[Dict[str, str]] = []
        if body_node:
            for a in body_node.find_all("a", href=True):
                href = a["href"].strip()
                link_text = a.get_text(strip=True)
                if href and not href.startswith("#") and not href.startswith("javascript:"):
                    links.append({"text": link_text, "url": urljoin(url, href)})

        # 6. Clean Text Content formatting (preserving headings and bullet lists)
        content_lines: List[str] = []
        if body_node:
            for elem in body_node.children:
                if isinstance(elem, Tag):
                    tag_name = elem.name
                    elem_text = elem.get_text(separator=" ", strip=True)
                    if not elem_text:
                        continue
                    if tag_name in ["h1", "h2"]:
                        content_lines.append(f"\n## {elem_text}\n")
                    elif tag_name in ["h3", "h4", "h5", "h6"]:
                        content_lines.append(f"\n### {elem_text}\n")
                    elif tag_name in ["ul", "ol"]:
                        for li in elem.find_all("li"):
                            li_text = li.get_text(separator=" ", strip=True)
                            if li_text:
                                content_lines.append(f"- {li_text}")
                    elif tag_name == "p":
                        content_lines.append(elem_text)
                    elif tag_name in ["pre", "code"]:
                        content_lines.append(f"```\n{elem_text}\n```")
                    else:
                        content_lines.append(elem_text)

        cleaned_content = "\n\n".join(content_lines).strip()
        if not cleaned_content and body_node:
            cleaned_content = body_node.get_text(separator="\n", strip=True)

        return ArticleData(
            article_id=article_id,
            url=url,
            title=title,
            category=category,
            folder=folder,
            breadcrumbs=breadcrumbs,
            content=cleaned_content,
            images=images,
            links=links,
            lastmod=lastmod,
            scraped_at=scraped_at,
        )
