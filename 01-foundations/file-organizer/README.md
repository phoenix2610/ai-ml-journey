# File Organizer

Sorts a directory into folders using rules you can edit without touching code.
The design goal was not "move files" — that is ten lines — but **being safe
enough to point at a real Downloads folder**:

- **Dry run is the default.** Nothing moves until you pass `--apply`.
- **Plan and apply are separate.** The preview you read is the exact list of
  moves that will run, because planning touches nothing on disk.
- **Every run is reversible.** Moves are journalled as they happen, so `undo`
  works even if the run was interrupted halfway.

## Use it

```bash
python -m organizer organize ~/Downloads              # preview only
python -m organizer organize ~/Downloads --apply      # do it
python -m organizer organize ~/Downloads -d ~/Sorted -r --duplicates=collect

python -m organizer duplicates ~/Pictures -r          # what is wasting space
python -m organizer undo                              # put the last run back
python -m organizer rules                             # what will go where
```

```
$ python -m organizer organize ~/Downloads
41 file(s) scanned in /home/you/Downloads
  invoice-april.pdf        ->  Documents/invoice-april.pdf
  Screenshot 2024-11-02.png ->  Images/Screenshots/2024-11/Screenshot 2024-11-02.png
  holiday.jpg              ->  Images/2023/holiday.jpg
  budget.xlsx              ->  Documents/Spreadsheets/budget.xlsx
  ...
  skip report.pdf: duplicate of invoice-april.pdf

38 move(s) planned across 7 folder(s)
dry run -- nothing was moved. re-run with --apply to do it.
```

## Rules

`rules.toml`, not Python. Highest `priority` wins; ties go to whichever rule is
declared first.

```toml
[[rule]]
name = "images"
folder = "Images/{year}"
extensions = ["jpg", "png", "webp"]

# Screenshots *are* images, but a higher priority pulls them out of the
# year folders into their own bucket.
[[rule]]
name = "screenshots"
folder = "Images/Screenshots/{year}-{month}"
patterns = ["Screenshot*", "Screen Shot*"]
priority = 20
```

Folder templates understand `{year}` `{month}` `{month_name}` `{day}` `{ext}`,
so one rule can fan a decade of photos into per-year folders. A rule can match
on extension, filename glob, size range, or any combination.

## The parts that are actually hard

**Name collisions.** Two `note.txt` files from different subfolders both want
the same destination. Resolution happens during *planning*, against a set of
already-claimed paths — checking `exists()` alone would let both plan onto the
same target, since neither is there yet. Counters go before the extension, so
`archive.tar.gz` becomes `archive.tar (2).gz` and stays a gzip file.

**Duplicate detection without hashing everything.** Two files can only be
identical if their sizes match, and size comes free from the `stat()` the walk
already did. So: bucket by size, then hash only inside buckets with more than
one member, streaming in 1 MiB chunks. On a photo library that is a handful of
hashes instead of thousands.

**Undo that survives a crash.** Each move is appended to a JSONL journal and
`fsync`ed the instant it succeeds, not batched at the end — so an interrupted
run loses at most the move in flight. Undo walks a batch *backwards*, because
replaying forward can restore a file into a directory that a later move is
about to reclaim.

## Layout

```
organizer/
├── rules.py      TOML -> Rule objects; matching and folder templating
├── scanner.py    walking, hashing, duplicate grouping
├── planner.py    Move/Plan; collision resolution; apply
├── journal.py    append-only move log; undo
└── cli.py        argparse front-end
```

## Tests

```bash
pytest -q          # 67 tests
```

Weighted toward the failure paths: collision counters, duplicate strategies,
a file that vanishes between plan and apply, an interrupted journal with a torn
final line, and undo refusing to overwrite a path the user has since reused.
