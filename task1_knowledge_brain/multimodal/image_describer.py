"""Safe screenshot download, Claude vision analysis, and content-aware caching."""
import base64
import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx

from task1_knowledge_brain.config import anthropic_model, load_project_environment
from .image_extractor import ExtractedImage

logger = logging.getLogger(__name__)
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "image_descriptions_cache.json"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 3
ALLOWED_IMAGE_HOSTS = {"s3.amazonaws.com", "support.brilliantdirectories.com"}
ALLOWED_IMAGE_SUFFIXES = (".freshdesk.com", ".brilliantdirectories.com")
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class ImageDescriber:
    """Adds visual descriptions when Claude is configured, otherwise uses explicit alt/context text."""

    def __init__(self, cache_path: Path = CACHE_PATH, model_name: Optional[str] = None):
        load_project_environment()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.cache_path.parent / "image_cache"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name or anthropic_model()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model_failure: Optional[str] = None
        self.cache: Dict[str, Any] = {}
        self.load_cache()

    def load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.cache = loaded
        except Exception as exc:
            logger.warning("Could not read image-description cache (%s)", type(exc).__name__)

    def save_cache(self) -> None:
        temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        try:
            temp_path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(self.cache_path)
        except Exception as exc:
            logger.warning("Could not persist image-description cache (%s)", type(exc).__name__)

    @staticmethod
    def get_cache_key(image_url: str) -> str:
        return hashlib.sha256(image_url.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _allowed_url(url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or parsed.username or parsed.password:
            return False
        return host in ALLOWED_IMAGE_HOSTS or any(host.endswith(suffix) for suffix in ALLOWED_IMAGE_SUFFIXES)

    def _download_image(self, image_url: str, prior: Optional[Dict[str, Any]]) -> Tuple[bytes, str, Dict[str, Optional[str]], bool]:
        """Download bounded image bytes and revalidate cached content via HTTP validators."""
        headers = {"User-Agent": "AITechConsultantBot/1.0 (support article screenshot analyzer)"}
        if prior and prior.get("etag"):
            headers["If-None-Match"] = prior["etag"]
        if prior and prior.get("last_modified"):
            headers["If-Modified-Since"] = prior["last_modified"]
        url = image_url
        with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0), follow_redirects=False, headers=headers) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                if not self._allowed_url(url):
                    raise ValueError("Image URL or redirect host is not on the allowlist")
                with client.stream("GET", url) as response:
                    if response.status_code == 304 and prior:
                        cached_path = Path(prior.get("local_path", ""))
                        if cached_path.is_file():
                            body = cached_path.read_bytes()
                            return body, prior["media_type"], prior, True
                    if response.status_code in {301, 302, 303, 307, 308}:
                        if redirect_count >= MAX_REDIRECTS:
                            raise ValueError("Too many image redirects")
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Image redirect omitted Location")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if media_type not in SUPPORTED_IMAGE_TYPES:
                        raise ValueError("Unsupported screenshot media type")
                    declared = response.headers.get("content-length")
                    if declared and int(declared) > MAX_IMAGE_BYTES:
                        raise ValueError("Screenshot exceeds the download size limit")
                    parts: List[bytes] = []
                    total = 0
                    for part in response.iter_bytes():
                        total += len(part)
                        if total > MAX_IMAGE_BYTES:
                            raise ValueError("Screenshot exceeds the download size limit")
                        parts.append(part)
                    body = b"".join(parts)
                    if not body:
                        raise ValueError("Screenshot response was empty")
                    metadata = {
                        "etag": response.headers.get("etag"),
                        "last_modified": response.headers.get("last-modified"),
                        "media_type": media_type,
                    }
                    return body, media_type, metadata, False
        raise ValueError("Image download did not complete")

    def _call_vision_model(self, image_bytes: bytes, media_type: str, title: str, category: str, folder: str) -> Tuple[str, Dict[str, int]]:
        payload = {
            "model": self.model_name,
            "max_tokens": 300,
            "system": "Describe only visual elements that are actually visible. Do not infer hidden behavior or invent UI labels. State when text is unreadable.",
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                 "data": base64.b64encode(image_bytes).decode("ascii")}},
                {"type": "text", "text": f"Describe useful visible interface details in this support article screenshot. Article: {title}. Section: {category} / {folder}. Focus on visible controls, labels, sequence, and state; do not guess."},
            ]}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        with httpx.Client(timeout=45.0) as client:
            response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
        description = "\n".join(c["text"] for c in result.get("content", []) if c.get("type") == "text").strip()
        if not description:
            raise ValueError("Anthropic returned no image description")
        usage = result.get("usage", {})
        return description, {k: int(usage[k]) for k in ("input_tokens", "output_tokens") if usage.get(k) is not None}

    @staticmethod
    def _alt_context(image: Any, title: str, category: str, folder: str) -> str:
        alt = image.alt if hasattr(image, "alt") else image.get("alt")
        caption = image.caption if hasattr(image, "caption") else image.get("caption")
        prefix = f"Screenshot in '{title}' ({category} > {folder})"
        if alt and len(alt.strip()) > 15:
            text = f"{prefix}: {alt.strip()}"
        else:
            text = f"Documentation screenshot for '{title}' in section '{category}' / '{folder}'."
            if alt:
                text += f" Label: {alt.strip()}."
        if caption:
            text += f" Context: {caption.strip()}"
        return text

    @staticmethod
    def _set_description(image: Any, description: str, source: str, content_hash: Optional[str]) -> None:
        if hasattr(image, "vision_description"):
            image.vision_description = description
            image.description_source = source
            image.content_hash = content_hash
        elif isinstance(image, dict):
            image["vision_description"] = description
            image["description_source"] = source
            image["content_hash"] = content_hash

    def describe_image(self, image: Any, article_title: str, category: str, folder: str) -> str:
        src = image.src if hasattr(image, "src") else image.get("src", "")
        key = self.get_cache_key(src)
        prior = self.cache.get(key)
        if isinstance(prior, str):  # preserve old cache entries as metadata-only, never as visual analysis
            prior = {"description": prior, "source": "alt_or_context_legacy"}
        if not self.api_key or self.model_failure:
            description = self._alt_context(image, article_title, category, folder)
            self._set_description(image, description, "alt_or_context", None)
            return description

        try:
            content, media_type, validators, unchanged = self._download_image(src, prior)
            content_hash = hashlib.sha256(content).hexdigest()
            if (prior and prior.get("content_hash") == content_hash and prior.get("description")
                    and prior.get("source") == "claude_vision"):
                description = prior["description"]
                source = "claude_vision"
            else:
                description, usage = self._call_vision_model(content, media_type, article_title, category, folder)
                source = "claude_vision"
                filename = f"{content_hash}.{mimetypes.guess_extension(media_type) or '.img'}"
                local_path = self.image_dir / filename
                if not local_path.exists():
                    local_path.write_bytes(content)
                self.cache[key] = {
                    "url": src, "content_hash": content_hash, "description": description,
                    "source": source, "local_path": str(local_path), **validators,
                    "article_title": article_title, "category": category, "folder": folder,
                    "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                }
                self.save_cache()
            self._set_description(image, description, source, content_hash)
            return description
        except Exception as exc:
            # One provider/credential failure disables subsequent image calls
            # for this instance; article text and alt context remain available.
            self.model_failure = type(exc).__name__
            logger.warning("Screenshot analysis unavailable (%s); retaining article and alt/context metadata", self.model_failure)
            description = self._alt_context(image, article_title, category, folder)
            # Keep a successfully downloaded image and its validators even when
            # analysis fails, so retries can use conditional HTTP requests.
            if "content" in locals() and "content_hash" in locals() and "validators" in locals():
                filename = f"{content_hash}.{mimetypes.guess_extension(media_type) or '.img'}"
                local_path = self.image_dir / filename
                try:
                    if not local_path.exists():
                        local_path.write_bytes(content)
                    self.cache[key] = {
                        "url": src, "content_hash": content_hash, "description": description,
                        "source": "alt_or_context", "analysis_error": self.model_failure,
                        "local_path": str(local_path), **validators,
                        "article_title": article_title, "category": category, "folder": folder,
                    }
                    self.save_cache()
                except OSError:
                    logger.warning("Could not cache downloaded screenshot bytes")
            self._set_description(image, description, "alt_or_context", None)
            return description

    def enrich_article_images(self, images: List[ExtractedImage], article_title: str, category: str, folder: str) -> List[ExtractedImage]:
        for image in images:
            self.describe_image(image, article_title, category, folder)
        return images
