"""A crawler that behaves itself: robots.txt, rate limits, retries, caching."""

from scraper.fetcher import FetchError, Fetcher, Response

__all__ = ["Fetcher", "Response", "FetchError"]
__version__ = "0.1.0"
