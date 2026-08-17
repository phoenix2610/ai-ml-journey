"""Writing records out.

CSV is the awkward one: records are dicts that may not share keys, and some
values are lists. So the header is the union of every key in first-seen order,
and list values are joined rather than repr'd -- `"a|b"` opens cleanly in a
spreadsheet, `"['a', 'b']"` does not.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

LIST_SEPARATOR = " | "


def collect_columns(records: Sequence[dict[str, Any]]) -> list[str]:
    """Union of all keys, preserving the order they were first seen."""
    columns: dict[str, None] = {}
    for record in records:
        for key in record:
            columns[key] = None
    return list(columns)


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return LIST_SEPARATOR.join(str(v) for v in value)
    if value is None:
        return ""
    return value


def write_csv(records: Iterable[dict[str, Any]], path: Path | str) -> int:
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = collect_columns(records)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({c: _flatten(record.get(c)) for c in columns})
    return len(records)


def write_jsonl(records: Iterable[dict[str, Any]], path: Path | str) -> int:
    """One JSON object per line. Streams, so memory stays flat on big crawls."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(records: Iterable[dict[str, Any]], path: Path | str) -> int:
    records = list(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(records)


WRITERS = {"csv": write_csv, "jsonl": write_jsonl, "json": write_json}


def write(records: Iterable[dict[str, Any]], path: Path | str, fmt: str = "csv") -> int:
    if fmt not in WRITERS:
        raise ValueError(f"unknown format {fmt!r}; try one of {', '.join(WRITERS)}")
    return WRITERS[fmt](records, path)


__all__ = ["write", "write_csv", "write_jsonl", "write_json", "collect_columns", "WRITERS"]
