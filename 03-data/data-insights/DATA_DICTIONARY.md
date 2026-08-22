# Data dictionary

## Raw (`data/raw/transactions.csv`)

Everything is read as `object`/text. Typing is the cleaning layer's job, so the
raw frame preserves exactly what the "export" contained — including the mess.

| Column | Raw type | Notes |
|---|---|---|
| `invoice_id` | text | `INV100123`. Cancellations are prefixed `C`. |
| `invoice_date` | text | **Two formats:** `2023-05-06 10:14:00` and `06/05/2023`. |
| `customer_id` | text | ~14% missing, as `""`, `N/A`, `null`, or absent. |
| `country` | text | Inconsistent case and padding: `"  UNITED KINGDOM "`. |
| `category` | text | One of six. Same casing/padding problem. |
| `product` | text | Product name; each belongs to exactly one category. |
| `quantity` | text | Integer. **Negative means a return.** |
| `unit_price` | text | May carry a thousands separator (`"1,234.56"`) or be `0.00`. |

## Cleaned (`sales` and `returns` frames)

`clean()` returns two frames with identical schemas. `sales` has strictly
positive quantities; `returns` strictly negative.

| Column | Type | Notes |
|---|---|---|
| `invoice_id` | `string` | |
| `invoice_date` | `datetime64[ns]` | Both raw formats resolved. Unparseable rows dropped. |
| `customer_id` | `Int64` | Nullable. `<NA>` = guest checkout. |
| `country` | `string` | Normalised: trimmed, whitespace collapsed, title-cased. |
| `category` | `string` | Six values after normalisation. |
| `product` | `string` | |
| `quantity` | `Float64` | Positive in `sales`, negative in `returns`. |
| `unit_price` | `Float64` | Separators stripped. `0.0` retained and flagged. |
| `revenue` | `Float64` | **Derived:** `quantity × unit_price`. Negative in `returns`. |
| `month` | `datetime64[ns]` | **Derived:** first day of the invoice month. |

## Values

**Countries** — India, United Kingdom, Germany, France, Australia, Japan

**Categories and products**

| Category | Products |
|---|---|
| Home | Cushion Cover, Photo Frame, Wall Clock, Throw Blanket |
| Garden | Plant Pot, Watering Can, Seed Kit, Trowel |
| Kitchen | Ceramic Mug, Tea Towel, Storage Jar, Chopping Board |
| Stationery | Notebook, Pen Set, Desk Pad, Sticker Sheet |
| Toys | Wooden Puzzle, Spinning Top, Toy Train, Building Blocks |
| Lighting | Table Lamp, String Lights, Candle Set, Lantern |

## Known limitations

These bound every conclusion in [FINDINGS.md](FINDINGS.md):

- **No cost or margin data.** All rankings are revenue, never profit.
- **No acquisition channel or marketing spend.** Retention-vs-acquisition
  arguments have no cost side.
- **13.7% of orders have no customer.** Counted in revenue, invisible to every
  per-customer metric. This biases repeat-rate *downward* in reality.
- **No return reasons.** Return rates are counted, never explained.
- **Two years only.** Enough to call Q4 recurring; not enough to forecast it.

## Null handling

These tokens are all treated as missing, case-insensitively, after trimming:

```
""   " "   "na"   "n/a"   "null"   "none"   "nan"   "-"   "?"
```
