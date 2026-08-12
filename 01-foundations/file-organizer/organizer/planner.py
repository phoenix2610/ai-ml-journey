"""Deciding every move before making any of them.

Planning is fully separated from applying. `build_plan` touches nothing on
disk, which is what makes `--dry-run` trustworthy: the preview you read is the
exact list of moves that will run, not an approximation of them.

Name collisions are resolved during planning against a `claimed` set, so two
files that would land on the same path get distinct names even though neither
exists at the destination yet.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from organizer.rules import FileInfo, RuleSet
from organizer.scanner import find_duplicates

DuplicateStrategy = Literal["keep", "skip", "collect"]


@dataclass(frozen=True)
class Move:
    source: Path
    destination: Path
    reason: str


@dataclass
class Plan:
    moves: list[Move] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.moves)

    def __bool__(self) -> bool:
        return bool(self.moves)

    @property
    def folders(self) -> set[Path]:
        return {m.destination.parent for m in self.moves}


def unique_destination(destination: Path, claimed: set[Path]) -> Path:
    """Return a path that neither exists on disk nor is already spoken for.

    ``report.pdf`` -> ``report (2).pdf`` -> ``report (3).pdf`` ...
    The suffix goes before the extension so file-type association survives.
    """
    if destination not in claimed and not destination.exists():
        return destination

    stem, suffix = destination.stem, destination.suffix
    n = 2
    while True:
        candidate = destination.with_name(f"{stem} ({n}){suffix}")
        if candidate not in claimed and not candidate.exists():
            return candidate
        n += 1


def build_plan(
    files: Iterable[FileInfo],
    rules: RuleSet,
    destination_root: Path | str,
    *,
    duplicates: DuplicateStrategy = "keep",
) -> Plan:
    """Work out where every file should go. Reads the filesystem, writes nothing."""
    destination_root = Path(destination_root)
    files = list(files)
    plan = Plan()
    claimed: set[Path] = set()

    # Everything after the first member of a duplicate group is a "copy".
    redundant: dict[Path, Path] = {}
    if duplicates != "keep":
        for group in find_duplicates(files).values():
            original = group[0].path
            for dup in group[1:]:
                redundant[dup.path] = original

    for info in files:
        source = info.path

        if source in redundant:
            if duplicates == "skip":
                plan.skipped.append((source, f"duplicate of {redundant[source].name}"))
                continue
            folder, reason = "Duplicates", f"duplicate of {redundant[source].name}"
        else:
            folder = rules.categorise(info)
            reason = f"rule -> {folder}"

        target = unique_destination(destination_root / folder / info.name, claimed)

        if target == source:
            plan.skipped.append((source, "already in place"))
            continue

        claimed.add(target)
        plan.moves.append(Move(source=source, destination=target, reason=reason))

    return plan


def apply_plan(plan: Plan, *, journal=None, on_move=None) -> tuple[int, list[tuple[Move, str]]]:
    """Execute a plan. Returns (moved_count, failures).

    Each move is journalled the instant it succeeds rather than at the end, so
    an interrupted run is still fully undoable. A failure on one file never
    stops the rest.
    """
    moved = 0
    failures: list[tuple[Move, str]] = []

    for move in plan.moves:
        try:
            move.destination.parent.mkdir(parents=True, exist_ok=True)
            # shutil.move handles the cross-filesystem case that os.rename cannot.
            shutil.move(str(move.source), str(move.destination))
        except OSError as exc:
            failures.append((move, str(exc)))
            continue

        moved += 1
        if journal is not None:
            journal.record(move.source, move.destination)
        if on_move is not None:
            on_move(move)

    return moved, failures


__all__ = ["Move", "Plan", "build_plan", "apply_plan", "unique_destination"]
