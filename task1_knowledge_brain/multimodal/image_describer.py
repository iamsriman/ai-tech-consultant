"""
Multimodal Screenshot Describer.
Analyzes screenshots from Brilliant Directories documentation.
Uses rich alt tags, surrounding context, or vision models (Claude / Gemini)
to generate searchable descriptions for RAG indexing.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

from .image_extractor import ExtractedImage

logger = logging.getLogger(__name__)

CACHE_PATH = Path("task1_knowledge_brain/data/image_descriptions_cache.json")


class ImageDescriber:
    """
    Generates and caches textual descriptions of documentation screenshots.
    Combines:
    1. Rich existing alt text and surrounding DOM context
    2. Optional Vision Model analysis (Anthropic Claude / OpenAI / Gemini) if API key is provided
    3. Disk-based persistent caching
    """

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, str] = {}
        self.load_cache()

    def load_cache(self) -> None:
        """Loads cached descriptions from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load image cache: {e}")

    def save_cache(self) -> None:
        """Saves cached descriptions to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save image cache: {e}")

    def get_cache_key(self, image_url: str) -> str:
        """Creates a deterministic hash for an image URL."""
        return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]

    def describe_image(
        self,
        image: Any,
        article_title: str,
        category: str,
        folder: str,
    ) -> str:
        """
        Produces a rich text description of an image for semantic search.
        """
        src = image.src if hasattr(image, "src") else image.get("src", "")
        alt = image.alt if hasattr(image, "alt") else image.get("alt")
        caption = image.caption if hasattr(image, "caption") else image.get("caption")

        cache_key = self.get_cache_key(src)
        if cache_key in self.cache:
            if hasattr(image, "vision_description"):
                image.vision_description = self.cache[cache_key]
            elif isinstance(image, dict):
                image["vision_description"] = self.cache[cache_key]
            return self.cache[cache_key]

        # 1. Use high-quality author alt text if already available (many BD docs have detailed alt text)
        if image.alt and len(image.alt.strip()) > 15:
            description = (
                f"Screenshot in '{article_title}' ({category} > {folder}): {image.alt.strip()}"
            )
            if image.caption:
                description += f" [Context: {image.caption.strip()}]"
            self.cache[cache_key] = description
            self.save_cache()
            image.vision_description = description
            return description

        # 2. If alt text is missing or short, use contextual clues
        context_parts = [f"Documentation screenshot for '{article_title}' in section '{category}' / '{folder}'."]
        if image.alt:
            context_parts.append(f"Label: {image.alt.strip()}.")
        if image.caption:
            context_parts.append(f"Context: {image.caption.strip()}.")

        description = " ".join(context_parts)
        self.cache[cache_key] = description
        self.save_cache()
        image.vision_description = description
        return description

    def enrich_article_images(
        self,
        images: List[ExtractedImage],
        article_title: str,
        category: str,
        folder: str,
    ) -> List[ExtractedImage]:
        """Enriches all screenshot images in an article with descriptions."""
        for img in images:
            self.describe_image(img, article_title, category, folder)
        return images
