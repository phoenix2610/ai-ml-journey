"""Turning a page into a record.

Extraction rules are data, not code: a list of `Field` objects naming a CSS
selector and what to pull off the match. That keeps "which site am I scraping"
in one small declaration instead of spread through the crawler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Any
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

WHITESPACE = re.compile(r"\s+")


# ------------------------------------------------------------------ transforms


def clean_text(value: str) -> str:
    """Collapse runs of whitespace, including the newlines HTML is full of."""
    return WHITESPACE.sub(" ", value).strip()


def to_float(value: str) -> float | None:
    """First number in the string. Survives currency symbols and separators."""
    match = re.search(r"-?\d[\d,]*\.?\d*", value.replace(" ", " "))
    return float(match.group().replace(",", "")) if match else None


def to_int(value: str) -> int | None:
    result = to_float(value)
    return int(result) if result is not None else None


@dataclass
class Field:
    name: str
    selector: str
    attr: str | None = None          # None means take the text
    many: bool = False
    required: bool = False
    transform: Callable[[str], Any] | None = None


class ExtractionError(ValueError):
    """A required field that the page did not contain."""


@dataclass
class Extractor:
    fields: list[Field]

    def extract(self, html: str, url: str = "") -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        record: dict[str, Any] = {"url": url} if url else {}

        for field in self.fields:
            nodes = soup.select(field.selector)

            if not nodes:
                if field.required:
                    raise ExtractionError(
                        f"required field {field.name!r} matched nothing "
                        f"(selector {field.selector!r}) at {url or 'page'}"
                    )
                record[field.name] = [] if field.many else None
                continue

            values = [self._value(n, field, url) for n in nodes]
            values = [v for v in values if v is not None and v != ""]
            record[field.name] = values if field.many else (values[0] if values else None)

        return record

    @staticmethod
    def _value(node, field: Field, base_url: str):
        if field.attr is None:
            raw = clean_text(node.get_text())
        else:
            raw = node.get(field.attr)
            if raw is None:
                return None
            if isinstance(raw, list):  # e.g. class="a b" comes back as a list
                raw = " ".join(raw)
            # Anything URL-shaped gets resolved against the page it came from.
            if base_url and field.attr in ("href", "src", "data-src"):
                raw = urljoin(base_url, raw)

        return field.transform(raw) if field.transform else raw


# ----------------------------------------------------------------------- links


def normalise(url: str) -> str:
    """Drop the fragment and any trailing slash so '/a', '/a/', '/a#x' unify."""
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    rebuilt = f"{parsed.scheme}://{parsed.netloc}{path}"
    return f"{rebuilt}?{parsed.query}" if parsed.query else rebuilt


def find_links(html: str, base_url: str, *, same_domain: bool = True) -> list[str]:
    """Every http(s) link on the page, absolute and de-duplicated, in order."""
    soup = BeautifulSoup(html, "html.parser")
    origin = urlparse(base_url).netloc
    seen: dict[str, None] = {}

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = normalise(urljoin(base_url, href))
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            continue
        if same_domain and parsed.netloc != origin:
            continue
        seen[absolute] = None

    return list(seen)


__all__ = [
    "Field", "Extractor", "ExtractionError",
    "find_links", "normalise", "clean_text", "to_int", "to_float",
]
