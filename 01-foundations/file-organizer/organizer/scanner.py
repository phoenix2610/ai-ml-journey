"""Walking the source tree and finding duplicate content.

Hashing is the expensive part, so it is avoided wherever possible: two files
can only be duplicates if their sizes match, and size comes free from the stat
call the walk already makes. Only inside a size group does anything get read,
and then in chunks so a 4 GB video does not become 4 GB of resident memory.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from organizer.rules import FileInfo

CHUNK = 1 << 20  # 1 MiB

# Never touch these, even when they sit in the source directory.
SKIP_DIRS = {".git", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".cache"}


def file_hash(path: Path, *, chunk: int = CHUNK) -> str:
    """Streaming BLAKE2b digest. Faster than SHA-256 and plenty for dedupe."""
    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def scan(
    source: Path | str,
    *,
    recursive: bool = False,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> Iterator[FileInfo]:
    """Yield a FileInfo for every regular file under ``source``."""
    source = Path(source)
    if not source.is_dir():
        raise NotADirectoryError(f"not a directory: {source}")

    walker = source.rglob("*") if recursive else source.glob("*")
    for path in walker:
        if not include_hidden and any(part.startswith(".") for part in path.relative_to(source).parts):
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_symlink() and not follow_symlinks:
            continue
        if not path.is_file():
            continue
        try:
            yield FileInfo.from_path(path)
        except OSError:
            # Vanished between the walk and the stat, or unreadable. Skip it.
            continue


def find_duplicates(files: Iterable[FileInfo]) -> dict[str, list[FileInfo]]:
    """Group files by content hash, returning only the groups with 2+ members.

    Two passes: bucket by size first (free), then hash only inside buckets that
    have more than one member.
    """
    by_size: dict[int, list[FileInfo]] = defaultdict(list)
    for info in files:
        by_size[info.size].append(info)

    groups: dict[str, list[FileInfo]] = defaultdict(list)
    for size, candidates in by_size.items():
        if len(candidates) < 2 or size == 0:
            continue
        for info in candidates:
            try:
                groups[file_hash(info.path)].append(info)
            except OSError:
                continue

    return {h: sorted(g, key=lambda i: i.path) for h, g in groups.items() if len(g) > 1}


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"  # pragma: no cover


__all__ = ["scan", "find_duplicates", "file_hash", "human_size", "SKIP_DIRS"]
