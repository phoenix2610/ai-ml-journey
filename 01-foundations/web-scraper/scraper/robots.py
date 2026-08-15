"""robots.txt: parsing it, and actually obeying it.

Python ships ``urllib.robotparser``, but it ignores ``Crawl-delay`` and gives
no way to see *why* a URL was refused. Both matter here, so this is a small
purpose-built parser.

The matching rule is longest-match-wins: between ``Disallow: /admin`` and
``Allow: /admin/public``, the longer pattern decides. That is what every major
crawler implements, and it is the only reading under which ``Allow`` can carve
an exception out of a broader ``Disallow``.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse


@dataclass
class Group:
    agents: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    disallow: list[str] = field(default_factory=list)
    crawl_delay: float | None = None


@dataclass
class Robots:
    groups: list[Group] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> "Robots":
        robots = cls()
        current: Group | None = None
        # Consecutive User-agent lines share one group; a rule line ends the run.
        agents_open = False

        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue

            field_name, _, value = line.partition(":")
            field_name = field_name.strip().lower()
            value = value.strip()

            if field_name == "sitemap":
                robots.sitemaps.append(value)
                continue

            if field_name == "user-agent":
                if current is None or not agents_open:
                    current = Group()
                    robots.groups.append(current)
                    agents_open = True
                current.agents.append(value.lower())
                continue

            if current is None:
                continue  # a rule before any User-agent line; ignore it
            agents_open = False

            if field_name == "disallow":
                current.disallow.append(value)
            elif field_name == "allow":
                current.allow.append(value)
            elif field_name == "crawl-delay":
                try:
                    current.crawl_delay = float(value)
                except ValueError:
                    pass

        return robots

    # ------------------------------------------------------------------ lookup

    def _group_for(self, user_agent: str) -> Group | None:
        """Most specific matching group; '*' is the fallback."""
        ua = user_agent.lower()
        wildcard = None
        for group in self.groups:
            for agent in group.agents:
                if agent == "*":
                    wildcard = wildcard or group
                elif agent in ua:
                    return group
        return wildcard

    def can_fetch(self, user_agent: str, url: str) -> bool:
        group = self._group_for(user_agent)
        if group is None:
            return True

        path = urlparse(url).path or "/"
        best_len, verdict = -1, True

        for rule in group.disallow:
            if rule and _matches(rule, path) and len(rule) > best_len:
                best_len, verdict = len(rule), False
            elif rule == "":
                # 'Disallow:' with an empty value explicitly allows everything.
                continue
        for rule in group.allow:
            if rule and _matches(rule, path) and len(rule) >= best_len:
                best_len, verdict = len(rule), True

        return verdict

    def crawl_delay(self, user_agent: str) -> float | None:
        group = self._group_for(user_agent)
        return group.crawl_delay if group else None


def _matches(pattern: str, path: str) -> bool:
    """robots.txt wildcards: '*' is any run, '$' anchors the end."""
    if "*" not in pattern and "$" not in pattern:
        return path.startswith(pattern)
    if pattern.endswith("$"):
        return fnmatch.fnmatchcase(path, pattern[:-1])
    return fnmatch.fnmatchcase(path, pattern + "*")


class RobotsCache:
    """Fetches and remembers robots.txt per host. Absent or broken means allow."""

    def __init__(self, fetcher, user_agent: str) -> None:
        self.fetcher = fetcher
        self.user_agent = user_agent
        self._cache: dict[str, Robots] = {}

    def _for_url(self, url: str) -> Robots:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in self._cache:
            try:
                response = self.fetcher.get(urljoin(host, "/robots.txt"))
                self._cache[host] = Robots.parse(response.text)
            except Exception:
                # No robots.txt, or unreachable: the standard says crawl freely.
                self._cache[host] = Robots()
        return self._cache[host]

    def can_fetch(self, url: str) -> bool:
        return self._for_url(url).can_fetch(self.user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        return self._for_url(url).crawl_delay(self.user_agent)


__all__ = ["Robots", "RobotsCache", "Group"]
