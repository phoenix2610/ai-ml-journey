# Data Insights

Cleaning a messy transaction log and reporting what it actually says — including
the parts it cannot say.

**[Read the findings →](FINDINGS.md)**

```bash
pip install -r requirements.txt
python run.py --dark      # numbers to stdout, 8 figures to figures/
pytest -q                 # 59 tests
```

## The dataset is generated, on purpose

A data project whose first step is *"download a 300 MB file that may have moved"*
is a project nobody can reproduce — including its author six months later. So
`acquire.py` generates the data deterministically from a seed, and injects the
six defect classes that real transaction exports actually contain:

| Injected defect | Rate | Mirrors |
|---|---|---|
| Duplicate rows | 2% | retried batch uploads |
| Missing `customer_id` | 14% | guest checkout |
| Case/whitespace variants | 20% | `"  UNITED KINGDOM "` vs `"United Kingdom"` |
| Mixed date formats | 12% | `2023-05-06` and `06/05/2023` in one column |
| Negative quantities | 3% | cancellations mixed in with sales |
| Thousands separators, zero prices | 3.5% | `"1,234.56"` as text; giveaways |

`python -m src.acquire --url <csv>` swaps in a real dataset — the pipeline does
not care which it gets. The benefit of the generated path is that **tests can
assert known-correct answers** instead of "whatever the file happens to say".

## Two rules the cleaning module is built around

**1. Every step is logged.**

```
  step                  rows   removed  note

  drop duplicates     12,000  -240 (2.0%)  identical rows from batch retries
  normalise text      12,000         -  trim, collapse spaces, title-case
  unify nulls         12,000         -  9 sentinel tokens -> NA
  parse dates         12,000         -  0 unparseable dates dropped
  require numerics    12,000         -  quantity and unit_price present
  split returns       11,628  -372 (3.1%)  372 return rows set aside

  12,240 rows in -> 11,628 out (95.0% retained)
  ! 212 rows have unit_price == 0 (giveaways, kept)
  ! 1,596 sales (13.7%) have no customer_id -- excluded from per-customer
    analysis, kept for revenue
```

A cleaning script that silently drops 40% of the data is indistinguishable from
one that works, right up until the conclusions are wrong. `clean()` returns the
frame **and** a report.

**2. Returns are separated, not deleted.**

Negative quantities are cancellations. Dropping them inflates revenue; leaving
them mixed in corrupts "units sold". They come back as a **separate frame**, so
the analysis nets them off deliberately rather than by accident.

## The date-parsing trap

One column, two formats. The obvious fix is `dayfirst=True` — and it silently
misreads `2023-05-06` as 6 June.

```python
parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
still_missing = parsed.isna() & text.notna()
parsed = parsed.fillna(pd.to_datetime(text.where(still_missing), dayfirst=True, errors="coerce"))
```

Strict ISO first, day-first only for what is left. There is a test pinning that
`2023-05-06` stays in May.

## Charts follow the data's job

| Question | Job | Form |
|---|---|---|
| Is revenue growing? | change over time | line + 3-month mean |
| Which categories carry it? | ranked magnitude | horizontal bars, sorted |
| When does demand peak? | polarity vs a baseline | diverging bars around 0 |
| How concentrated is revenue? | cumulative share | Lorenz curve vs equality |

Colour is assigned by role, not taste: categorical hues in a **fixed order** so
a series keeps its colour across every figure; a **single hue** for magnitude
(bar length already encodes it — a rainbow adds colour without information); a
**neutral grey midpoint** for the diverging chart so "average" reads as nothing.
The palette was checked for colour-vision-deficiency separation (worst adjacent
pair ΔE 24.7 protan, well clear of the ≥8 floor) and ≥3:1 contrast on both
surfaces. Dark mode is a **selected** set of steps, not an inverted light mode.

Deliberately not used: pie charts, and dual-axis charts — two y-scales let you
draw any conclusion you like.

## Layout

```
src/
├── acquire.py     generate or fetch; inject realistic mess
├── clean.py       the pipeline + CleaningReport
├── analyse.py     eight tidy-frame questions, no printing, no plotting
├── visualise.py   four figures, light and dark
└── theme.py       the colour system and matplotlib rcParams
run.py             the whole thing, one command
```

`analyse.py` never prints and never plots — that keeps the numbers testable and
lets one function feed a chart, a report, and an assertion.

## Tests

59 tests, weighted toward the cleaning traps: every null sentinel, mixed date
formats in one column, idempotency (re-cleaning clean data changes nothing),
determinism, all-guest input, and empty input.
