import pytest

from organizer.journal import Journal
from organizer.planner import apply_plan, build_plan
from organizer.rules import Rule, RuleSet
from organizer.scanner import scan


@pytest.fixture
def rules():
    return RuleSet(rules=[Rule("images", "Images", extensions=frozenset({"png"}))])


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "journal.jsonl", batch="batch-1")


def test_empty_journal_reads_as_empty(journal):
    assert journal.entries() == []


def test_record_appends(journal, tmp_path):
    journal.record(tmp_path / "a", tmp_path / "b")
    journal.record(tmp_path / "c", tmp_path / "d")
    assert len(journal.entries()) == 2


def test_entries_carry_paths(journal, tmp_path):
    journal.record(tmp_path / "a", tmp_path / "b")
    entry = journal.entries()[0]
    assert entry.src == tmp_path / "a"
    assert entry.dst == tmp_path / "b"
    assert entry.batch == "batch-1"


def test_torn_final_line_is_tolerated(journal, tmp_path):
    journal.record(tmp_path / "a", tmp_path / "b")
    with journal.path.open("a") as fh:
        fh.write('{"batch": "x", "sou')  # truncated by a crash
    assert len(journal.entries()) == 1


def test_undo_restores_files(tmp_path, rules, journal):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "a.png").write_text("img")

    apply_plan(build_plan(scan(src), rules, dest), journal=journal)
    assert not (src / "a.png").exists()

    restored, problems = journal.undo()
    assert (restored, problems) == (1, [])
    assert (src / "a.png").read_text() == "img"


def test_undo_clears_the_batch(tmp_path, rules, journal):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "a.png").write_text("img")
    apply_plan(build_plan(scan(src), rules, dest), journal=journal)
    journal.undo()
    assert journal.entries() == []


def test_undo_only_affects_the_named_batch(tmp_path, rules):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    jpath = tmp_path / "journal.jsonl"

    (src / "a.png").write_text("one")
    apply_plan(build_plan(scan(src), rules, dest), journal=Journal(jpath, batch="b1"))

    (src / "b.png").write_text("two")
    apply_plan(build_plan(scan(src), rules, dest), journal=Journal(jpath, batch="b2"))

    restored, _ = Journal(jpath).undo("b1")
    assert restored == 1
    assert (src / "a.png").exists()
    assert not (src / "b.png").exists()  # b2 untouched
    assert Journal(jpath).batches() == ["b2"]


def test_undo_defaults_to_the_most_recent_batch(tmp_path, rules):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    jpath = tmp_path / "journal.jsonl"

    (src / "a.png").write_text("one")
    apply_plan(build_plan(scan(src), rules, dest), journal=Journal(jpath, batch="b1"))
    (src / "b.png").write_text("two")
    apply_plan(build_plan(scan(src), rules, dest), journal=Journal(jpath, batch="b2"))

    Journal(jpath).undo()
    assert (src / "b.png").exists()
    assert not (src / "a.png").exists()


def test_undo_refuses_when_the_original_path_is_occupied(tmp_path, rules, journal):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "a.png").write_text("img")
    apply_plan(build_plan(scan(src), rules, dest), journal=journal)

    (src / "a.png").write_text("something new")  # user put a file back
    restored, problems = journal.undo()
    assert restored == 0
    assert "occupied" in problems[0]
    assert (src / "a.png").read_text() == "something new"


def test_undo_reports_a_file_that_went_missing(tmp_path, rules, journal):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.mkdir()
    (src / "a.png").write_text("img")
    apply_plan(build_plan(scan(src), rules, dest), journal=journal)

    (dest / "Images" / "a.png").unlink()
    restored, problems = journal.undo()
    assert restored == 0
    assert "missing" in problems[0]


def test_undo_on_empty_journal_is_not_an_error(journal):
    restored, problems = journal.undo()
    assert restored == 0
    assert "empty" in problems[0]
