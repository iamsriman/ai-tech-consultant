"""
Multimodal package for Task 1: Knowledge Brain.
Handles article extraction, image extraction, and vision summaries.
"""

from .image_extractor import ExtractedImage, ArticleData, ArticleParser
from .image_describer import ImageDescriber

__all__ = [
    "ExtractedImage",
    "ArticleData",
    "ArticleParser",
    "ImageDescriber",
]
