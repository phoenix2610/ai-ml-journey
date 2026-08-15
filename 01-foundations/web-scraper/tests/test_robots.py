import pytest

from scraper.fetcher import FetchError, Response
from scraper.robots import Robots, RobotsCache

BASIC = """
User-agent: *
Disallow: /private
Disallow: /tmp
Crawl-delay: 2

User-agent: badbot
Disallow: /
"""


def test_allows_by_default():
    assert Robots.parse("").can_fetch("mybot", "http://x.com/anything")


def test_disallowed_prefix():
    r = Robots.parse(BASIC)
    assert not r.can_fetch("mybot", "http://x.com/private/data")
    assert r.can_fetch("mybot", "http://x.com/public")


def test_specific_agent_group_wins_over_wildcard():
    r = Robots.parse(BASIC)
    assert not r.can_fetch("badbot/1.0", "http://x.com/public")
    assert r.can_fetch("mybot/1.0", "http://x.com/public")


def test_crawl_delay_is_read():
    assert Robots.parse(BASIC).crawl_delay("mybot") == 2.0


def test_invalid_crawl_delay_is_ignored():
    assert Robots.parse("User-agent: *\nCrawl-delay: soon").crawl_delay("mybot") is None


def test_comments_and_blank_lines():
    r = Robots.parse("# hello\n\nUser-agent: *  # everyone\nDisallow: /x\n")
    assert not r.can_fetch("mybot", "http://x.com/x")


def test_empty_disallow_allows_everything():
    assert Robots.parse("User-agent: *\nDisallow:").can_fetch("mybot", "http://x.com/any")


def test_longest_match_wins_so_allow_can_carve_an_exception():
    r = Robots.parse("User-agent: *\nDisallow: /admin\nAllow: /admin/public")
    assert not r.can_fetch("mybot", "http://x.com/admin/secret")
    assert r.can_fetch("mybot", "http://x.com/admin/public/page")


def test_wildcard_pattern():
    r = Robots.parse("User-agent: *\nDisallow: /*.pdf")
    assert not r.can_fetch("mybot", "http://x.com/docs/report.pdf")
    assert r.can_fetch("mybot", "http://x.com/docs/report.html")


def test_end_anchor():
    r = Robots.parse("User-agent: *\nDisallow: /page$")
    assert not r.can_fetch("mybot", "http://x.com/page")
    assert r.can_fetch("mybot", "http://x.com/page/sub")


def test_consecutive_user_agents_share_one_group():
    r = Robots.parse("User-agent: a\nUser-agent: b\nDisallow: /x")
    assert not r.can_fetch("a", "http://x.com/x")
    assert not r.can_fetch("b", "http://x.com/x")


def test_sitemaps_are_collected():
    r = Robots.parse("Sitemap: http://x.com/sitemap.xml\nUser-agent: *\nDisallow:")
    assert r.sitemaps == ["http://x.com/sitemap.xml"]


def test_rule_before_any_user_agent_is_ignored():
    assert Robots.parse("Disallow: /x").can_fetch("mybot", "http://x.com/x")


# ------------------------------------------------------------------ RobotsCache


class StubFetcher:
    def __init__(self, body=BASIC, fail=False):
        self.body = body
        self.fail = fail
        self.calls = 0

    def get(self, url):
        self.calls += 1
        if self.fail:
            raise FetchError("nope")
        return Response(url, 200, self.body)


def test_cache_consults_robots():
    cache = RobotsCache(StubFetcher(), "mybot")
    assert not cache.can_fetch("http://x.com/private/x")
    assert cache.can_fetch("http://x.com/open")


def test_robots_is_fetched_once_per_host():
    f = StubFetcher()
    cache = RobotsCache(f, "mybot")
    cache.can_fetch("http://x.com/a")
    cache.can_fetch("http://x.com/b")
    assert f.calls == 1


def test_missing_robots_means_allow_everything():
    cache = RobotsCache(StubFetcher(fail=True), "mybot")
    assert cache.can_fetch("http://x.com/anything")


def test_crawl_delay_via_cache():
    assert RobotsCache(StubFetcher(), "mybot").crawl_delay("http://x.com/") == 2.0
