"""
Polite Rate Limiter and Retry Handler for Scraping.
Ensures crawlers do not overload the Brilliant Directories documentation server.
Includes configurable delays, jitter, backoff, and polite header rotation.
"""

import asyncio
import logging
import random
import time
from typing import Callable, Optional, TypeVar, Any
import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PoliteRateLimiter:
    """
    Manages request timing, adaptive backoff, and concurrency control.
    """

    def __init__(
        self,
        min_delay: float = 0.5,
        max_delay: float = 1.0,
        max_retries: int = 4,
        backoff_factor: float = 2.0,
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.last_request_time: float = 0.0
        self.total_requests: int = 0
        self.total_throttles: int = 0

    def wait(self) -> None:
        """Synchronous polite delay between consecutive requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        target_delay = random.uniform(self.min_delay, self.max_delay)

        if elapsed < target_delay:
            sleep_duration = target_delay - elapsed
            time.sleep(sleep_duration)

        self.last_request_time = time.time()
        self.total_requests += 1

    async def wait_async(self) -> None:
        """Asynchronous polite delay between consecutive requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        target_delay = random.uniform(self.min_delay, self.max_delay)

        if elapsed < target_delay:
            sleep_duration = target_delay - elapsed
            await asyncio.sleep(sleep_duration)

        self.last_request_time = time.time()
        self.total_requests += 1

    def execute_with_retry(self, func: Callable[[], T]) -> T:
        """
        Executes a callable with automatic polite backoff on HTTP 429, 503, or network errors.
        """
        retries = 0
        while True:
            self.wait()
            try:
                return func()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 500, 502, 503, 504) and retries < self.max_retries:
                    retries += 1
                    self.total_throttles += 1
                    delay = (self.backoff_factor ** retries) + random.uniform(0.5, 1.5)
                    logger.warning(
                        f"HTTP {status} encountered. Backing off for {delay:.2f}s (retry {retries}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as e:
                if retries < self.max_retries:
                    retries += 1
                    delay = (self.backoff_factor ** retries) + random.uniform(0.5, 1.0)
                    logger.warning(
                        f"Network/Timeout error: {e}. Backing off for {delay:.2f}s (retry {retries}/{self.max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    raise
