"""An append-only record of what was moved, so any run can be reversed.

Stored as JSON Lines: one move per line, appended and flushed immediately. A
crash mid-run therefore loses at most the move that was in flight, and every
move that already happened is still undoable.

Undo walks a batch backwards. Order matters -- reversing the moves in the
order they were made can put a file back into a directory a later move is
about to reclaim.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_JOURNAL = Path.home() / ".local" / "share" / "file-organizer" / "journal.jsonl"


@dataclass
class Entry:
    batch: str
    timestamp: str
    source: str
    destination: str

    @property
    def src(self) -> Path:
        return Path(self.source)

    @property
    def dst(self) -> Path:
        return Path(self.destination)


class Journal:
    def __init__(self, path: Path | str = DEFAULT_JOURNAL, batch: str | None = None) -> None:
        self.path = Path(path)
        self.batch = batch or datetime.now().strftime("%Y%m%dT%H%M%S")

    # ------------------------------------------------------------------ write

    def record(self, source: Path, destination: Path) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "batch": self.batch,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source": str(source),
            "destination": str(destination),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ------------------------------------------------------------------- read

    def entries(self) -> list[Entry]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(Entry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # tolerate a torn final line
        return out

    def batches(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.entries():
            seen[e.batch] = None
        return list(seen)

    # ------------------------------------------------------------------- undo

    def undo(self, batch: str | None = None) -> tuple[int, list[str]]:
        """Move a batch's files back. Defaults to the most recent batch."""
        entries = self.entries()
        if not entries:
            return 0, ["journal is empty -- nothing to undo"]

        batch = batch or entries[-1].batch
        targets = [e for e in entries if e.batch == batch]
        if not targets:
            return 0, [f"no such batch: {batch}"]

        restored = 0
        problems: list[str] = []

        for entry in reversed(targets):
            if not entry.dst.exists():
                problems.append(f"missing, cannot restore: {entry.destination}")
                continue
            if entry.src.exists():
                problems.append(f"original path is occupied: {entry.source}")
                continue
            try:
                entry.src.parent.mkdir(parents=True, exist_ok=True)
                entry.dst.rename(entry.src)
                restored += 1
            except OSError as exc:
                problems.append(f"{entry.destination}: {exc}")

        if restored:
            self._forget(batch)
        return restored, problems

    def _forget(self, batch: str) -> None:
        """Drop a batch from the journal once it has been reversed."""
        keep = [e for e in self.entries() if e.batch != batch]
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for e in keep:
                fh.write(
                    json.dumps(
                        {
                            "batch": e.batch,
                            "timestamp": e.timestamp,
                            "source": e.source,
                            "destination": e.destination,
                        }
                    )
                    + "\n"
                )
        tmp.replace(self.path)


__all__ = ["Journal", "Entry", "DEFAULT_JOURNAL"]
