import json

import pytest

from scraper.crawl import Crawler, CrawlConfig
from scraper.export import collect_columns, write, write_csv, write_jsonl
from scraper.extract import Extractor, Field
from scraper.fetcher import FetchError, Response

SITE = {
    "http://t.test/": '<h1>Home</h1><a href="/a">a</a><a href="/b">b</a>',
    "http://t.test/a": '<h1>A</h1><a href="/c">c</a><a href="/">home</a>',
    "http://t.test/b": '<h1>B</h1>',
    "http://t.test/c": '<h1>C</h1>',
}


class SiteFetcher:
    """Serves a dict of url -> html; anything else is a 404."""

    def __init__(self, pages=None):
        self.pages = pages if pages is not None else SITE
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        if url not in self.pages:
            raise FetchError(f"{url}: HTTP 404")
        return Response(url, 200, self.pages[url])


class BlockingRobots:
    def __init__(self, blocked):
        self.blocked = blocked

    def can_fetch(self, url):
        return url not in self.blocked


@pytest.fixture
def extractor():
    return Extractor([Field("heading", "h1")])


def crawl(extractor, **kwargs):
    fetcher = SiteFetcher(kwargs.pop("pages", None))
    robots = kwargs.pop("robots", None)
    config = CrawlConfig(seeds=["http://t.test/"], **kwargs)
    crawler = Crawler(fetcher, extractor, config, robots=robots)
    return crawler, crawler.run()


# ----------------------------------------------------------------------- crawl


def test_visits_the_seed(extractor):
    _, records = crawl(extractor, max_depth=0)
    assert [r["heading"] for r in records] == ["Home"]


def test_follows_links_one_level(extractor):
    _, records = crawl(extractor, max_depth=1)
    assert sorted(r["heading"] for r in records) == ["A", "B", "Home"]


def test_depth_limit_is_respected(extractor):
    _, records = crawl(extractor, max_depth=2)
    assert sorted(r["heading"] for r in records) == ["A", "B", "C", "Home"]


def test_max_pages_stops_the_crawl(extractor):
    crawler, records = crawl(extractor, max_depth=5, max_pages=2)
    assert len(records) == 2
    assert crawler.stats.visited == 2


def test_a_page_is_never_visited_twice(extractor):
    # '/a' links back to '/', which is already in the seen set.
    fetcher = SiteFetcher()
    crawler = Crawler(fetcher, extractor, CrawlConfig(seeds=["http://t.test/"], max_depth=3))
    crawler.run()
    assert len(fetcher.requested) == len(set(fetcher.requested)) == 4


def test_breadth_first_order(extractor):
    fetcher = SiteFetcher()
    order = []
    crawler = Crawler(
        fetcher,
        extractor,
        CrawlConfig(seeds=["http://t.test/"], max_depth=2),
        on_page=lambda url, depth: order.append(depth),
    )
    crawler.run()
    assert order == sorted(order)  # depths never decrease


def test_follow_pattern_filters_the_frontier(extractor):
    _, records = crawl(extractor, max_depth=2, follow=r"/a$")
    assert sorted(r["heading"] for r in records) == ["A", "Home"]


def test_collect_pattern_filters_extraction(extractor):
    crawler, records = crawl(extractor, max_depth=1, collect=r"/[ab]$")
    assert sorted(r["heading"] for r in records) == ["A", "B"]
    assert crawler.stats.visited == 3  # home was fetched, just not collected


def test_fetch_failures_are_recorded_not_raised(extractor):
    pages = {"http://t.test/": '<h1>Home</h1><a href="/gone">x</a>'}
    crawler, records = crawl(extractor, max_depth=1, pages=pages)
    assert len(records) == 1
    assert crawler.stats.failed == 1
    assert "404" in crawler.stats.errors[0]


def test_robots_blocked_pages_are_skipped(extractor):
    fetcher = SiteFetcher()
    crawler = Crawler(
        fetcher,
        extractor,
        CrawlConfig(seeds=["http://t.test/"], max_depth=1),
        robots=BlockingRobots({"http://t.test/a"}),
    )
    records = crawler.run()
    assert "A" not in [r["heading"] for r in records]
    assert crawler.stats.skipped_robots == 1


# ---------------------------------------------------------------------- export


RECORDS = [
    {"url": "u1", "title": "One", "tags": ["x", "y"]},
    {"url": "u2", "title": "Two", "extra": 5},
]


def test_collect_columns_is_the_union_in_first_seen_order():
    assert collect_columns(RECORDS) == ["url", "title", "tags", "extra"]


def test_write_csv_has_a_header_and_a_row_per_record(tmp_path):
    path = tmp_path / "out.csv"
    assert write_csv(RECORDS, path) == 2
    lines = path.read_text().splitlines()
    assert lines[0] == "url,title,tags,extra"
    assert len(lines) == 3


def test_csv_joins_list_values(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(RECORDS, path)
    assert "x | y" in path.read_text()


def test_csv_renders_missing_keys_as_empty(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(RECORDS, path)
    assert path.read_text().splitlines()[2].endswith(",5")


def test_write_jsonl_round_trips(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(RECORDS, path)
    parsed = [json.loads(line) for line in path.read_text().splitlines()]
    assert parsed == RECORDS


def test_write_creates_parent_directories(tmp_path):
    write(RECORDS, tmp_path / "deep" / "nested" / "out.csv", "csv")
    assert (tmp_path / "deep" / "nested" / "out.csv").exists()


def test_unknown_format_raises():
    with pytest.raises(ValueError, match="unknown format"):
        write(RECORDS, "x.txt", "xml")
