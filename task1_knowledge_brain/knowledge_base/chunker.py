"""
Hierarchy-Aware Markdown & Context-Injected Chunker for Task 1: Knowledge Brain.
Splits articles into semantic chunks of ~600 tokens, prepends hierarchical context
(Article title, Category, Folder, URL), and attaches screenshot descriptions.
"""

from dataclasses import dataclass, asdict
import re
from typing import Dict, List, Optional
from task1_knowledge_brain.multimodal.image_extractor import ArticleData, ExtractedImage


@dataclass
class DocumentChunk:
    chunk_id: str
    article_id: str
    title: str
    category: str
    folder: str
    url: str
    chunk_index: int
    total_chunks: int
    text: str
    has_screenshot: bool = False
    screenshot_urls: str = ""

    def to_metadata(self) -> Dict[str, str | int | float | bool]:
        """Returns a flat dictionary strictly compatible with ChromaDB metadata requirements."""
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "category": self.category,
            "folder": self.folder,
            "url": self.url,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "has_screenshot": self.has_screenshot,
            "screenshot_urls": self.screenshot_urls[:500],  # Chroma metadata length safety
        }


class HierarchyAwareChunker:
    """
    Chunks articles while preserving markdown sections, hierarchy context,
    and multimodal screenshot descriptions.
    """

    def __init__(self, target_chunk_size: int = 600, overlap: int = 60):
        # target_chunk_size in approximate words/tokens (~4 chars per word)
        self.target_tokens = target_chunk_size
        self.overlap_tokens = overlap

    def _split_into_sections(self, content: str) -> List[str]:
        """Splits markdown content by headings or double newlines."""
        # Split on headers (## or ###) but retain header text
        raw_sections = re.split(r"(?=\n#{1,4}\s+)", content)
        cleaned_sections = [s.strip() for s in raw_sections if s.strip()]
        if not cleaned_sections:
            cleaned_sections = [p.strip() for p in content.split("\n\n") if p.strip()]
        return cleaned_sections or [content]

    def chunk_article(self, article: ArticleData) -> List[DocumentChunk]:
        """
        Processes an ArticleData object into a list of DocumentChunks with
        injected context headers and screenshot references.
        """
        # Context header attached to every chunk
        context_header = (
            f"[Article: {article.title} | Category: {article.category} | "
            f"Folder: {article.folder} | URL: {article.url}]\n\n"
        )

        # Collect screenshot descriptions from article images
        screenshot_lines: List[str] = []
        screenshot_urls: List[str] = []
        for img in article.images:
            img_src = img.src if hasattr(img, "src") else img.get("src", "")
            img_alt = img.alt if hasattr(img, "alt") else img.get("alt", "")
            img_desc = getattr(img, "vision_description", None) or (img.get("vision_description") if isinstance(img, dict) else None)
            
            desc_text = img_desc or img_alt or "Documentation Screenshot"
            if img_src:
                screenshot_lines.append(f"- Screenshot: {desc_text} (Image: {img_src})")
                screenshot_urls.append(img_src)

        screenshot_block = ""
        if screenshot_lines:
            screenshot_block = "\n\n[Associated UI Screenshots]:\n" + "\n".join(screenshot_lines[:4])

        sections = self._split_into_sections(article.content)
        chunks_text_list: List[str] = []
        current_chunk_words: List[str] = []

        for section in sections:
            words = section.split()
            if not words:
                continue

            # If adding this section exceeds target tokens, seal current chunk
            if current_chunk_words and (len(current_chunk_words) + len(words)) > self.target_tokens:
                chunk_str = " ".join(current_chunk_words)
                chunks_text_list.append(chunk_str)

                # Keep overlap from previous chunk
                overlap_words = current_chunk_words[-self.overlap_tokens:] if len(current_chunk_words) > self.overlap_tokens else []
                current_chunk_words = overlap_words + words
            else:
                current_chunk_words.extend(words)

        # Seal final remaining chunk
        if current_chunk_words:
            chunks_text_list.append(" ".join(current_chunk_words))

        # If article content is empty, generate at least 1 chunk with title and breadcrumbs
        if not chunks_text_list:
            chunks_text_list = [f"Article: {article.title}. Section: {article.category} > {article.folder}"]

        total_chunks = len(chunks_text_list)
        document_chunks: List[DocumentChunk] = []

        has_screenshots = len(screenshot_urls) > 0
        joined_screenshots = "; ".join(screenshot_urls)

        for idx, text in enumerate(chunks_text_list):
            chunk_id = f"{article.article_id}_c{idx}"

            # Append screenshots to first chunk or any chunk describing UI steps
            full_chunk_text = context_header + text
            if idx == 0 and screenshot_block:
                full_chunk_text += screenshot_block

            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                article_id=str(article.article_id),
                title=article.title,
                category=article.category,
                folder=article.folder,
                url=article.url,
                chunk_index=idx,
                total_chunks=total_chunks,
                text=full_chunk_text,
                has_screenshot=has_screenshots,
                screenshot_urls=joined_screenshots,
            )
            document_chunks.append(doc_chunk)

        return document_chunks
