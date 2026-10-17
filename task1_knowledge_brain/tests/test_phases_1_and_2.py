"""
Comprehensive Integration Test for Phase 1 & Phase 2.
Tests:
- Phase 1: SitemapParser, PoliteRateLimiter, CrawlCheckpoint
- Phase 2: ArticleParser, ImageExtractor, ImageDescriber, PoliteCrawler, and Disk Persistence
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from task1_knowledge_brain.scraper import (
    SitemapParser,
    PoliteRateLimiter,
    CrawlCheckpoint,
    PoliteCrawler,
)
from task1_knowledge_brain.multimodal import (
    ArticleParser,
    ImageDescriber,
    ArticleData,
    ExtractedImage,
)


def run_tests():
    print("=" * 65)
    print("        RUNNING TEST SUITE: PHASE 1 & PHASE 2 VALIDATION        ")
    print("=" * 65)

    passed_tests = 0
    total_tests = 7

    # -------------------------------------------------------------
    # TEST 1: Sitemap Parser
    # -------------------------------------------------------------
    print("\n[TEST 1/7] Testing SitemapParser against live BD sitemap...")
    try:
        parser = SitemapParser()
        entries = parser.parse()
        articles = parser.get_articles_only(entries)
        categories = parser.get_categories_and_folders(entries)

        assert len(entries) > 1000, f"Expected >1000 total URLs, got {len(entries)}"
        assert len(articles) > 1000, f"Expected >1000 articles, got {len(articles)}"
        assert len(categories) > 0, "Expected categories to be detected"
        assert articles[0].article_id is not None, "Article ID must not be None"
        assert articles[0].slug is not None, "Article slug must not be None"

        print(f"  --> PASSED: Discovered {len(entries)} total entries ({len(articles)} articles).")
        print(f"      Sample article: ID={articles[0].article_id}, slug='{articles[0].slug[:35]}...'")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 1: {e}")

    # -------------------------------------------------------------
    # TEST 2: Polite Rate Limiter
    # -------------------------------------------------------------
    print("\n[TEST 2/7] Testing PoliteRateLimiter timing & pacing...")
    try:
        limiter = PoliteRateLimiter(min_delay=0.2, max_delay=0.35)
        start_time = time.time()
        for _ in range(3):
            limiter.wait()
        elapsed = time.time() - start_time

        assert elapsed >= 0.4, f"Rate limiter too fast: {elapsed:.3f}s"
        assert limiter.total_requests == 3, f"Expected 3 requests, got {limiter.total_requests}"
        print(f"  --> PASSED: Paced 3 requests across {elapsed:.3f}s with polite delays.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 2: {e}")

    # -------------------------------------------------------------
    # TEST 3: Checkpoint Persistence & Recovery
    # -------------------------------------------------------------
    print("\n[TEST 3/7] Testing CrawlCheckpoint (State saving, resume & recovery)...")
    try:
        test_ckpt_path = Path("task1_knowledge_brain/data/test_checkpoint.json")
        if test_ckpt_path.exists():
            test_ckpt_path.unlink()

        ckpt = CrawlCheckpoint(test_ckpt_path)
        test_url = "https://support.brilliantdirectories.com/support/solutions/articles/99999-test-article"
        
        # Mark completed
        ckpt.mark_completed(test_url, article_id="99999", lastmod="2026-10-17T00:00:00Z")
        assert ckpt.is_completed(test_url), "URL should be marked completed"
        
        # Test reload from disk
        ckpt_reloaded = CrawlCheckpoint(test_ckpt_path)
        assert ckpt_reloaded.is_completed(test_url), "Reloaded checkpoint must preserve state"
        
        # Progress stats
        stats = ckpt_reloaded.get_progress(total=10)
        assert stats["completed"] == 1, f"Expected 1 completed, got {stats['completed']}"
        assert stats["remaining"] == 9, f"Expected 9 remaining, got {stats['remaining']}"
        assert stats["percent"] == 10.0, f"Expected 10.0%, got {stats['percent']}"

        # Clean up test checkpoint
        if test_ckpt_path.exists():
            test_ckpt_path.unlink()

        print(f"  --> PASSED: State persisted and reloaded successfully (progress: {stats['percent']}%).")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 3: {e}")

    # -------------------------------------------------------------
    # TEST 4: Article Content & Structure Extraction
    # -------------------------------------------------------------
    print("\n[TEST 4/7] Testing ArticleParser on live sample article...")
    try:
        article_parser = ArticleParser()
        sample_entry = articles[0]
        crawler = PoliteCrawler(rate_limiter=PoliteRateLimiter(min_delay=0.1, max_delay=0.2))
        html = crawler.fetch_page_html(sample_entry.loc)
        
        data = article_parser.parse(
            html=html,
            url=sample_entry.loc,
            article_id=sample_entry.article_id,
            lastmod=sample_entry.lastmod,
        )

        assert len(data.title) > 0, "Article title must not be empty"
        assert len(data.category) > 0, "Category must not be empty"
        assert len(data.folder) > 0, "Folder must not be empty"
        assert len(data.content) > 50, f"Content too short ({len(data.content)} chars)"
        assert len(data.breadcrumbs) >= 2, "Breadcrumbs should have at least 2 levels"

        print(f"  --> PASSED: Parsed article '{data.title[:40]}...'")
        print(f"      Category: '{data.category}' | Folder: '{data.folder}'")
        print(f"      Content length: {len(data.content)} characters")
        print(f"      Breadcrumbs: {' > '.join(data.breadcrumbs)}")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 4: {e}")

    # -------------------------------------------------------------
    # TEST 5: Multimodal Screenshot Extraction & Alt Context
    # -------------------------------------------------------------
    print("\n[TEST 5/7] Testing Screenshot & Multimodal Extraction on article with images...")
    try:
        # Article 5000558363 has screenshots
        image_article_url = "https://support.brilliantdirectories.com/support/solutions/articles/5000558363-emails-email-accounts"
        html_img = crawler.fetch_page_html(image_article_url)
        img_data = article_parser.parse(
            html=html_img,
            url=image_article_url,
            article_id="5000558363",
        )

        assert len(img_data.images) > 0, f"Expected screenshots in article, found {len(img_data.images)}"
        screenshot = img_data.images[0]
        assert screenshot.is_screenshot is True, "Image should be flagged as screenshot"
        assert screenshot.src.startswith("http"), f"Image src must be valid URL: {screenshot.src}"

        print(f"  --> PASSED: Detected {len(img_data.images)} screenshots in article.")
        print(f"      Image 1 URL: {screenshot.src[:65]}...")
        print(f"      Image 1 Alt: '{screenshot.alt}'")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 5: {e}")

    # -------------------------------------------------------------
    # TEST 6: ImageDescriber and Semantic Search Enrichment
    # -------------------------------------------------------------
    print("\n[TEST 6/7] Testing ImageDescriber (Description generation & caching)...")
    try:
        describer = ImageDescriber()
        desc = describer.describe_image(
            screenshot,
            article_title=img_data.title,
            category=img_data.category,
            folder=img_data.folder,
        )

        assert len(desc) > 20, f"Description too short: '{desc}'"
        assert "Screenshot" in desc or "Admin" in desc or img_data.title in desc

        # Verify disk caching of image description
        cache_key = describer.get_cache_key(screenshot.src)
        assert cache_key in describer.cache, "Image description must be cached"

        print(f"  --> PASSED: Generated rich search description:")
        print(f"      '{desc[:120]}...'")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 6: {e}")

    # -------------------------------------------------------------
    # TEST 7: Resumable Crawler & Disk Storage
    # -------------------------------------------------------------
    print("\n[TEST 7/7] Testing End-to-End Resumable Crawler & Disk Persistence...")
    try:
        # Load all articles currently saved
        saved_articles = crawler.load_all_saved_articles()
        assert len(saved_articles) >= 3, f"Expected at least 3 saved articles, found {len(saved_articles)}"

        # Verify full schema of first saved article
        first = saved_articles[0]
        assert isinstance(first, ArticleData)
        assert first.article_id is not None
        assert first.url is not None
        assert first.title is not None
        assert first.content is not None

        # Verify checkpoint resumability: already completed articles must be skipped
        pending = crawler.checkpoint.filter_pending([first.url, "https://nonexistent-url-test.com"])
        assert first.url not in pending, "Completed URL must be excluded from pending list"
        assert "https://nonexistent-url-test.com" in pending, "Uncrawled URL must remain in pending list"

        print(f"  --> PASSED: Verified {len(saved_articles)} articles stored on disk.")
        print(f"      Resumable filter correctly skipped completed article {first.article_id}.")
        passed_tests += 1
    except Exception as e:
        print(f"  --> FAILED in Test 7: {e}")

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    if passed_tests == total_tests:
        print(f"   ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! (PHASES 1 & 2 OPERATIONAL)")
    else:
        print(f"   COMPLETED: {passed_tests}/{total_tests} TESTS PASSED.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tests()
