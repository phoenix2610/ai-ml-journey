# Polite Web Scraper

A crawler built around the constraint that makes scraping hard in practice —
**not getting blocked**. Fetching a page is one line of `requests`; doing it a
thousand times without being rude, and without re-downloading everything when
the run dies at page 900, is the actual project.

```bash
python -m scraper https://books.toscrape.com/ \
    --field 'title:h3 a@title' \
    --field 'price:p.price_color:float' \
    --field 'rating:p.star-rating@class' \
    --follow 'catalogue/' --max-pages 40 --cache .cache \
    -o books.csv
```

## What "polite" means here

| | |
|---|---|
| **robots.txt** | Parsed and obeyed, including `Crawl-delay`. Fetched once per host. |
| **Rate limiting** | Minimum gap between requests, tracked **per host** with jitter. |
| **Retries** | Exponential backoff, capped at 30s, only on 408/425/429/5xx. |
| **Caching** | Content-addressed on disk, so a re-run costs zero requests. |
| **Identification** | A real User-Agent, not a browser impersonation. |

The rate limiter is per-host rather than global because being polite to one
site should not slow down a crawl spanning twenty. The jitter matters more than
it looks — a crawler hitting a host on an exact metronome is trivially
fingerprinted, and a fleet of them retrying in lockstep is a thundering herd.

## Design: the transport is injected

```python
class Transport(Protocol):
    def get(self, url, headers, timeout) -> Response: ...
```

`Fetcher` depends on that protocol, not on `requests`. Two things fall out:
the entire 77-test suite runs offline against a scripted fake, and swapping in
`httpx` or an async transport later touches one class instead of the crawler.

## robots.txt, and why not `urllib.robotparser`

The stdlib parser ignores `Crawl-delay` and cannot tell you *why* a URL was
refused. Both matter, so `robots.py` is a small purpose-built parser.

The rule that trips people up is **longest-match-wins**:

```
Disallow: /admin
Allow:    /admin/public
```

`/admin/secret` is blocked; `/admin/public/page` is not — the longer pattern
decides. Under any other reading, `Allow` could never carve an exception out of
a broader `Disallow`, which is the only reason it exists.

## Crawl order

Breadth-first, deliberately. A crawl cut short by `--max-pages` should have
covered the site broadly rather than tunnelled into one branch. Every URL is
normalised before entering the frontier, so `/a`, `/a/`, and `/a#section` are
one page instead of three.

Two independent regex filters, because they answer different questions:

- `--follow` — which links are worth **enqueueing** (navigation)
- `--collect` — which pages are worth **extracting a record from** (content)

On a paginated catalogue you follow `catalogue/page-\d+` but collect from
product URLs only.

## Field specs

Extraction rules are data, so a whole scrape fits on the command line:

```
--field 'title:h3 a@title'        take the @title attribute
--field 'price:p.price:float'     pull the first number out of "£51.77"
--field 'tags:ul.tags li[]'       [] means collect every match
```

`href`/`src` values are resolved against the page they came from, so relative
links come out absolute.

## Layout

```
scraper/
├── fetcher.py    transport, retries, rate limiting, cache
├── robots.py     robots.txt parsing and per-host caching
├── extract.py    CSS-selector rules -> records; link discovery
├── crawl.py      BFS frontier with depth/page limits
├── export.py     CSV / JSONL / JSON writers
└── cli.py        argparse front-end
```

## Tests

```bash
pip install -r requirements.txt
pytest -q          # 77 tests, no network access
```

Concentrated on the things that are painful to debug live: which status codes
retry and which do not, that the per-host limiter does not serialise unrelated
hosts, longest-match robots resolution, that a torn cache entry reads as a
miss, and that a 404 mid-crawl is recorded rather than fatal.

## Scrape responsibly

Check a site's Terms of Service, prefer an official API, keep `--delay` at 1s
or higher, and cache aggressively so you only ever fetch a page once.
