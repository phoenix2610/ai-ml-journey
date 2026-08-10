"""Declarative rules: given a file, decide which folder it belongs in.

Rules live in TOML rather than in Python so that changing where screenshots go
does not mean editing code. A rule matches on extension, filename glob, or
size; the highest-priority match wins, and ties break toward the rule declared
first. Destination folders are templates, so a single rule can fan a decade of
photos into per-year folders.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MONTHS = (
    "January February March April May June July "
    "August September October November December"
).split()


@dataclass(frozen=True)
class FileInfo:
    """Everything a rule is allowed to look at."""

    path: Path
    size: int
    mtime: datetime

    @classmethod
    def from_path(cls, path: Path) -> "FileInfo":
        stat = path.stat()
        return cls(
            path=path,
            size=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime),
        )

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def ext(self) -> str:
        """Lower-case extension without the dot. Empty string if there is none."""
        return self.path.suffix.lstrip(".").lower()

    def template_vars(self) -> dict[str, str]:
        return {
            "ext": self.ext or "none",
            "year": f"{self.mtime.year:04d}",
            "month": f"{self.mtime.month:02d}",
            "month_name": MONTHS[self.mtime.month - 1],
            "day": f"{self.mtime.day:02d}",
        }


@dataclass
class Rule:
    name: str
    folder: str
    extensions: frozenset[str] = frozenset()
    patterns: tuple[str, ...] = ()
    min_size: int = 0
    max_size: int | None = None
    priority: int = 0

    def matches(self, info: FileInfo) -> bool:
        if info.size < self.min_size:
            return False
        if self.max_size is not None and info.size > self.max_size:
            return False

        # A rule with neither extensions nor patterns is a pure size rule and
        # matches anything that passed the size gate above.
        if not self.extensions and not self.patterns:
            return True

        if self.extensions and info.ext in self.extensions:
            return True
        return any(fnmatch.fnmatch(info.name, p) for p in self.patterns)

    def destination(self, info: FileInfo) -> str:
        """Render the folder template for this file."""
        try:
            return self.folder.format(**info.template_vars())
        except KeyError as exc:
            raise RuleError(f"rule {self.name!r}: unknown template variable {exc}") from None


class RuleError(ValueError):
    """A rules file that cannot be understood."""


@dataclass
class RuleSet:
    rules: list[Rule] = field(default_factory=list)
    fallback: str = "Other"

    def categorise(self, info: FileInfo) -> str:
        """Return the destination folder for ``info``, or the fallback."""
        best: Rule | None = None
        for rule in self.rules:
            if rule.matches(info) and (best is None or rule.priority > best.priority):
                best = rule
        return best.destination(info) if best else self.fallback

    @classmethod
    def from_toml(cls, path: Path | str) -> "RuleSet":
        path = Path(path)
        if not path.exists():
            raise RuleError(f"rules file not found: {path}")
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "RuleSet":
        rules = []
        for i, entry in enumerate(raw.get("rule", [])):
            missing = {"name", "folder"} - entry.keys()
            if missing:
                raise RuleError(f"rule #{i + 1} is missing {', '.join(sorted(missing))}")
            rules.append(
                Rule(
                    name=entry["name"],
                    folder=entry["folder"],
                    extensions=frozenset(e.lower().lstrip(".") for e in entry.get("extensions", [])),
                    patterns=tuple(entry.get("patterns", [])),
                    min_size=int(entry.get("min_size", 0)),
                    max_size=entry.get("max_size"),
                    priority=int(entry.get("priority", 0)),
                )
            )
        return cls(rules=rules, fallback=raw.get("fallback", "Other"))


__all__ = ["FileInfo", "Rule", "RuleSet", "RuleError"]
