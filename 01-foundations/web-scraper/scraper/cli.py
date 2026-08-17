"""Command-line interface.

Field specs are given inline so a scrape can be described entirely on the
command line:

    --field 'title:h2 a@title'      text of h2 a, or the @title attribute
    --field 'price:p.price:float'   run the value through a transform
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scraper.crawl import Crawler, CrawlConfig
from scraper.export import WRITERS, write
from scraper.extract import Extractor, Field, to_float, to_int
from scraper.fetcher import DEFAULT_UA, Fetcher
from scraper.robots import RobotsCache

TRANSFORMS = {"float": to_float, "int": to_int, "text": None}


def parse_field(spec: str) -> Field:
    """``name:selector[@attr][:transform]`` -> Field."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"bad field spec {spec!r}; expected name:selector[@attr][:transform]"
        )

    name, selector = parts[0], parts[1]
    transform_name = parts[2] if len(parts) > 2 else "text"

    if transform_name not in TRANSFORMS:
        raise argparse.ArgumentTypeError(
            f"unknown transform {transform_name!r}; try {', '.join(TRANSFORMS)}"
        )

    attr = None
    many = False
    if selector.endswith("[]"):
        selector, many = selector[:-2], True
    if "@" in selector:
        selector, attr = selector.rsplit("@", 1)

    return Field(
        name=name,
        selector=selector.strip(),
        attr=attr,
        many=many,
        transform=TRANSFORMS[transform_name],
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scrape", description="A crawler that behaves itself.")
    p.add_argument("seeds", nargs="+", help="one or more starting URLs")
    p.add_argument(
        "-f", "--field", dest="fields", action="append", type=parse_field, default=[],
        metavar="SPEC", help="name:selector[@attr][:transform], repeatable",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("out.csv"))
    p.add_argument("--format", choices=sorted(WRITERS), default=None)
    p.add_argument("-n", "--max-pages", type=int, default=50)
    p.add_argument("-d", "--max-depth", type=int, default=2)
    p.add_argument("--follow", metavar="REGEX", help="only enqueue links matching this")
    p.add_argument("--collect", metavar="REGEX", help="only extract from URLs matching this")
    p.add_argument("--offsite", action="store_true", help="allow leaving the seed domain")
    p.add_argument("--delay", type=float, default=1.0, help="seconds between requests per host")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--user-agent", default=DEFAULT_UA)
    p.add_argument("--cache", type=Path, metavar="DIR", help="cache responses here")
    p.add_argument("--ignore-robots", action="store_true", help="do not consult robots.txt")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.fields:
        # With no fields the crawl still runs; you just get a list of URLs.
        args.fields = [Field("title", "title"), Field("h1", "h1")]

    fmt = args.format or (args.output.suffix.lstrip(".") if args.output.suffix else "csv")
    if fmt not in WRITERS:
        print(f"error: cannot infer format from {args.output.name!r}; pass --format", file=sys.stderr)
        return 2

    fetcher = Fetcher(
        user_agent=args.user_agent,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
        cache_dir=args.cache,
    )
    robots = None if args.ignore_robots else RobotsCache(fetcher, args.user_agent)

    # A site's own Crawl-delay beats our default when it asks for more.
    if robots is not None:
        requested = robots.crawl_delay(args.seeds[0])
        if requested and requested > args.delay:
            if not args.quiet:
                print(f"robots.txt asks for {requested}s between requests; honouring that")
            fetcher.limiter.delay = requested

    config = CrawlConfig(
        seeds=args.seeds,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        same_domain=not args.offsite,
        follow=args.follow,
        collect=args.collect,
    )

    def progress(url: str, depth: int) -> None:
        print(f"  [{depth}] {url}", file=sys.stderr)

    crawler = Crawler(
        fetcher,
        Extractor(args.fields),
        config,
        robots=robots,
        on_page=None if args.quiet else progress,
    )

    try:
        records = crawler.run()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    written = write(records, args.output, fmt)
    stats = crawler.stats

    print(f"\n{written} record(s) -> {args.output}")
    print(
        f"visited {stats.visited}, extracted {stats.extracted}, "
        f"failed {stats.failed}, robots-skipped {stats.skipped_robots}"
    )
    print(
        f"network: {fetcher.stats['fetched']} fetched, "
        f"{fetcher.stats['cached']} from cache, {fetcher.stats['retried']} retried"
    )

    for err in stats.errors[:5]:
        print(f"  ! {err}", file=sys.stderr)
    if len(stats.errors) > 5:
        print(f"  ... and {len(stats.errors) - 5} more errors", file=sys.stderr)

    return 0


__all__ = ["main", "build_parser", "parse_field"]
