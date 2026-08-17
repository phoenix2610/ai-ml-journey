"""The crawl loop: a breadth-first frontier with hard stopping conditions.

Breadth-first rather than depth-first on purpose. A crawl that is cut short by
`--max-pages` should have covered the site broadly, not tunnelled into one
branch, and BFS gives that for free.

Every URL is normalised before it enters the frontier, so `/a`, `/a/`, and
`/a#section` are one page rather than three.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from scraper.extract import Extractor, ExtractionError, find_links, normalise
from scraper.fetcher import FetchError, Fetcher


@dataclass
class CrawlConfig:
    seeds: list[str]
    max_pages: int = 50
    max_depth: int = 2
    same_domain: bool = True
    follow: str | None = None    # regex: which links to enqueue
    collect: str | None = None   # regex: which pages to extract a record from


@dataclass
class CrawlStats:
    visited: int = 0
    extracted: int = 0
    skipped_robots: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class Crawler:
    def __init__(
        self,
        fetcher: Fetcher,
        extractor: Extractor,
        config: CrawlConfig,
        *,
        robots=None,
        on_page: Callable[[str, int], None] | None = None,
    ) -> None:
        self.fetcher = fetcher
        self.extractor = extractor
        self.config = config
        self.robots = robots
        self.on_page = on_page
        self.stats = CrawlStats()

    def run(self) -> list[dict[str, Any]]:
        return list(self.iter_records())

    def iter_records(self) -> Iterator[dict[str, Any]]:
        cfg = self.config
        follow_re = re.compile(cfg.follow) if cfg.follow else None
        collect_re = re.compile(cfg.collect) if cfg.collect else None

        frontier: deque[tuple[str, int]] = deque(
            (normalise(seed), 0) for seed in cfg.seeds
        )
        seen: set[str] = {url for url, _ in frontier}

        while frontier and self.stats.visited < cfg.max_pages:
            url, depth = frontier.popleft()

            if self.robots is not None and not self.robots.can_fetch(url):
                self.stats.skipped_robots += 1
                continue

            try:
                response = self.fetcher.get(url)
            except FetchError as exc:
                self.stats.failed += 1
                self.stats.errors.append(str(exc))
                continue

            self.stats.visited += 1
            if self.on_page is not None:
                self.on_page(url, depth)

            if collect_re is None or collect_re.search(url):
                try:
                    record = self.extractor.extract(response.text, url)
                except ExtractionError as exc:
                    self.stats.errors.append(str(exc))
                else:
                    self.stats.extracted += 1
                    yield record

            if depth >= cfg.max_depth:
                continue

            for link in find_links(response.text, url, same_domain=cfg.same_domain):
                if link in seen:
                    continue
                if follow_re is not None and not follow_re.search(link):
                    continue
                seen.add(link)
                frontier.append((link, depth + 1))


__all__ = ["Crawler", "CrawlConfig", "CrawlStats"]
