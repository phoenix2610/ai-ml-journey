import pytest

from scraper.extract import (
    ExtractionError,
    Extractor,
    Field,
    clean_text,
    find_links,
    normalise,
    to_float,
    to_int,
)

PAGE = """
<html><body>
  <h1 class="title">  The   Great\n  Book </h1>
  <span class="price">£12.99</span>
  <span class="stock">In stock (7 available)</span>
  <ul class="tags"><li>fiction</li><li>classic</li></ul>
  <img id="cover" src="/media/cover.jpg">
  <a href="/next">next</a>
  <a href="/next/">next again</a>
  <a href="https://other.com/x">offsite</a>
  <a href="mailto:a@b.com">mail</a>
  <a href="#top">anchor</a>
</body></html>
"""

BASE = "http://books.test/catalogue/page-1"


# ------------------------------------------------------------------ transforms


def test_clean_text_collapses_whitespace():
    assert clean_text("  a \n\n  b  ") == "a b"


@pytest.mark.parametrize(
    "raw,expected",
    [("£12.99", 12.99), ("1,234.5", 1234.5), ("-7", -7.0), ("no digits", None)],
)
def test_to_float(raw, expected):
    assert to_float(raw) == expected


def test_to_int_truncates():
    assert to_int("In stock (7 available)") == 7


# ------------------------------------------------------------------- extractor


@pytest.fixture
def extractor():
    return Extractor(
        [
            Field("title", "h1.title"),
            Field("price", "span.price", transform=to_float),
            Field("stock", "span.stock", transform=to_int),
            Field("tags", "ul.tags li", many=True),
            Field("cover", "#cover", attr="src"),
        ]
    )


def test_text_field_is_cleaned(extractor):
    assert extractor.extract(PAGE, BASE)["title"] == "The Great Book"


def test_transform_is_applied(extractor):
    record = extractor.extract(PAGE, BASE)
    assert record["price"] == 12.99
    assert record["stock"] == 7


def test_many_collects_every_match(extractor):
    assert extractor.extract(PAGE, BASE)["tags"] == ["fiction", "classic"]


def test_attribute_urls_are_made_absolute(extractor):
    assert extractor.extract(PAGE, BASE)["cover"] == "http://books.test/media/cover.jpg"


def test_url_is_recorded(extractor):
    assert extractor.extract(PAGE, BASE)["url"] == BASE


def test_missing_optional_field_is_none():
    e = Extractor([Field("nope", ".does-not-exist")])
    assert e.extract(PAGE, BASE)["nope"] is None


def test_missing_optional_many_field_is_empty_list():
    e = Extractor([Field("nope", ".does-not-exist", many=True)])
    assert e.extract(PAGE, BASE)["nope"] == []


def test_missing_required_field_raises():
    e = Extractor([Field("nope", ".does-not-exist", required=True)])
    with pytest.raises(ExtractionError, match="required field"):
        e.extract(PAGE, BASE)


def test_missing_attribute_yields_none():
    e = Extractor([Field("alt", "#cover", attr="alt")])
    assert e.extract(PAGE, BASE)["alt"] is None


# ----------------------------------------------------------------- normalise


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("http://x.com/a", "http://x.com/a"),
        ("http://x.com/a/", "http://x.com/a"),
        ("http://x.com/a#frag", "http://x.com/a"),
        ("http://x.com/", "http://x.com/"),
        ("http://x.com/a?b=1", "http://x.com/a?b=1"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


# ---------------------------------------------------------------- find_links


def test_links_are_absolute():
    assert "http://books.test/next" in find_links(PAGE, BASE)


def test_offsite_links_excluded_by_default():
    assert not any("other.com" in u for u in find_links(PAGE, BASE))


def test_offsite_links_can_be_included():
    assert any("other.com" in u for u in find_links(PAGE, BASE, same_domain=False))


def test_non_http_schemes_are_dropped():
    assert not any(u.startswith("mailto") for u in find_links(PAGE, BASE, same_domain=False))


def test_duplicates_after_normalisation_are_collapsed():
    # '/next' and '/next/' are the same page.
    assert find_links(PAGE, BASE).count("http://books.test/next") == 1


def test_bare_fragment_is_not_a_link():
    assert normalise(BASE) not in [u for u in find_links(PAGE, BASE) if u.endswith("#top")]
