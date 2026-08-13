"""Command-line interface.

Default is a dry run. Nothing moves until you pass ``--apply``, because a tool
that rearranges a home directory should make the destructive path the one you
have to ask for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from organizer.journal import DEFAULT_JOURNAL, Journal
from organizer.planner import apply_plan, build_plan
from organizer.rules import RuleError, RuleSet
from organizer.scanner import find_duplicates, human_size, scan

DEFAULT_RULES = Path(__file__).resolve().parent.parent / "rules.toml"


def _load_rules(path: Path) -> RuleSet:
    try:
        return RuleSet.from_toml(path)
    except RuleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def cmd_organize(args: argparse.Namespace) -> int:
    rules = _load_rules(args.rules)
    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination or source).expanduser().resolve()

    try:
        files = list(scan(source, recursive=args.recursive, include_hidden=args.hidden))
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not files:
        print(f"no files found in {source}")
        return 0

    plan = build_plan(files, rules, destination, duplicates=args.duplicates)

    print(f"{len(files)} file(s) scanned in {source}")
    if not plan:
        print("nothing to do -- everything is already where it should be")
        return 0

    width = max(len(m.source.name) for m in plan.moves)
    for move in plan.moves:
        rel = move.destination.relative_to(destination)
        print(f"  {move.source.name:<{width}}  ->  {rel}")

    for path, why in plan.skipped:
        print(f"  skip {path.name}: {why}")

    if not args.apply:
        print(f"\n{len(plan)} move(s) planned across {len(plan.folders)} folder(s)")
        print("dry run -- nothing was moved. re-run with --apply to do it.")
        return 0

    journal = Journal(args.journal)
    moved, failures = apply_plan(plan, journal=journal)
    print(f"\nmoved {moved} file(s)")

    for move, err in failures:
        print(f"  failed {move.source.name}: {err}", file=sys.stderr)

    if moved:
        print(f"undo with:  {Path(sys.argv[0]).name} undo --batch {journal.batch}")
    return 1 if failures else 0


def cmd_duplicates(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    groups = find_duplicates(scan(source, recursive=args.recursive, include_hidden=args.hidden))

    if not groups:
        print("no duplicates found")
        return 0

    wasted = 0
    for group in groups.values():
        print(f"\n{human_size(group[0].size)} x{len(group)}")
        for i, info in enumerate(group):
            marker = "keep" if i == 0 else "dupe"
            print(f"  [{marker}] {info.path}")
        wasted += group[0].size * (len(group) - 1)

    print(f"\n{len(groups)} group(s), {human_size(wasted)} recoverable")
    return 0


def cmd_undo(args: argparse.Namespace) -> int:
    journal = Journal(args.journal)
    restored, problems = journal.undo(args.batch)
    print(f"restored {restored} file(s)")
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 1 if problems and not restored else 0


def cmd_rules(args: argparse.Namespace) -> int:
    rules = _load_rules(args.rules)
    print(f"{len(rules.rules)} rule(s), fallback -> {rules.fallback}\n")
    for rule in sorted(rules.rules, key=lambda r: -r.priority):
        bits = []
        if rule.extensions:
            bits.append(" ".join(sorted(rule.extensions)))
        if rule.patterns:
            bits.append(" ".join(rule.patterns))
        if rule.min_size:
            bits.append(f">={human_size(rule.min_size)}")
        print(f"  {rule.priority:>4}  {rule.name:<14} -> {rule.folder}")
        if bits:
            print(f"        {'; '.join(bits)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="organize", description=__doc__)
    p.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL, help="journal file location")
    sub = p.add_subparsers(dest="command", required=True)

    def add_scan_flags(sp):
        sp.add_argument("-r", "--recursive", action="store_true", help="descend into subfolders")
        sp.add_argument("--hidden", action="store_true", help="include dotfiles")

    org = sub.add_parser("organize", help="sort files into folders")
    org.add_argument("source", help="folder to organize")
    org.add_argument("-d", "--destination", help="where sorted folders go (default: in place)")
    org.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    org.add_argument(
        "--duplicates",
        choices=("keep", "skip", "collect"),
        default="keep",
        help="what to do with identical files (default: keep)",
    )
    org.add_argument("--apply", action="store_true", help="actually move files")
    add_scan_flags(org)
    org.set_defaults(func=cmd_organize)

    dup = sub.add_parser("duplicates", help="report identical files")
    dup.add_argument("source")
    add_scan_flags(dup)
    dup.set_defaults(func=cmd_duplicates)

    und = sub.add_parser("undo", help="reverse a previous run")
    und.add_argument("--batch", help="batch id (default: most recent)")
    und.set_defaults(func=cmd_undo)

    rul = sub.add_parser("rules", help="show the loaded rules")
    rul.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    rul.set_defaults(func=cmd_rules)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


__all__ = ["main", "build_parser"]
