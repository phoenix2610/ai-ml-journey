import pytest

from organizer.planner import apply_plan, build_plan, unique_destination
from organizer.rules import Rule, RuleSet
from organizer.scanner import scan


@pytest.fixture
def rules():
    return RuleSet(
        rules=[
            Rule("images", "Images", extensions=frozenset({"png", "jpg"})),
            Rule("docs", "Documents", extensions=frozenset({"pdf", "txt"})),
        ],
        fallback="Other",
    )


@pytest.fixture
def src(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    return d


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "sorted"


# ------------------------------------------------------------ unique_destination


def test_free_path_is_returned_unchanged(tmp_path):
    target = tmp_path / "a.txt"
    assert unique_destination(target, set()) == target


def test_existing_path_gets_a_counter(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x")
    assert unique_destination(target, set()).name == "a (2).txt"


def test_claimed_path_gets_a_counter_even_when_absent(tmp_path):
    target = tmp_path / "a.txt"
    assert unique_destination(target, {target}).name == "a (2).txt"


def test_counter_increments_past_multiple_collisions(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "a (2).txt").write_text("x")
    assert unique_destination(tmp_path / "a.txt", set()).name == "a (3).txt"


def test_counter_goes_before_the_extension(tmp_path):
    target = tmp_path / "archive.tar.gz"
    target.write_text("x")
    # .stem of 'archive.tar.gz' is 'archive.tar', so the suffix is preserved.
    assert unique_destination(target, set()).name == "archive.tar (2).gz"


# ------------------------------------------------------------------ build_plan


def test_files_are_routed_by_rule(src, dest, rules):
    (src / "a.png").write_text("img")
    (src / "b.pdf").write_text("doc")
    plan = build_plan(scan(src), rules, dest)
    routes = {m.source.name: m.destination.parent.name for m in plan.moves}
    assert routes == {"a.png": "Images", "b.pdf": "Documents"}


def test_unmatched_file_uses_fallback(src, dest, rules):
    (src / "x.qqq").write_text("?")
    plan = build_plan(scan(src), rules, dest)
    assert plan.moves[0].destination.parent.name == "Other"


def test_planning_writes_nothing_to_disk(src, dest, rules):
    (src / "a.png").write_text("img")
    build_plan(scan(src), rules, dest)
    assert not dest.exists()
    assert (src / "a.png").exists()


def test_two_same_named_files_do_not_collide(src, dest, rules):
    sub = src / "sub"
    sub.mkdir()
    (src / "note.txt").write_text("one")
    (sub / "note.txt").write_text("two")
    plan = build_plan(scan(src, recursive=True), rules, dest)
    targets = {m.destination.name for m in plan.moves}
    assert targets == {"note.txt", "note (2).txt"}


def test_plan_reports_target_folders(src, dest, rules):
    (src / "a.png").write_text("img")
    (src / "b.pdf").write_text("doc")
    plan = build_plan(scan(src), rules, dest)
    assert {p.name for p in plan.folders} == {"Images", "Documents"}


# ------------------------------------------------------------------ duplicates


def test_duplicates_kept_by_default(src, dest, rules):
    (src / "a.png").write_text("same")
    (src / "b.png").write_text("same")
    plan = build_plan(scan(src), rules, dest)
    assert len(plan) == 2


def test_duplicates_skipped(src, dest, rules):
    (src / "a.png").write_text("same")
    (src / "b.png").write_text("same")
    plan = build_plan(scan(src), rules, dest, duplicates="skip")
    assert len(plan) == 1
    assert len(plan.skipped) == 1
    assert "duplicate of" in plan.skipped[0][1]


def test_duplicates_collected(src, dest, rules):
    (src / "a.png").write_text("same")
    (src / "b.png").write_text("same")
    plan = build_plan(scan(src), rules, dest, duplicates="collect")
    folders = sorted(m.destination.parent.name for m in plan.moves)
    assert folders == ["Duplicates", "Images"]


# ------------------------------------------------------------------ apply_plan


def test_apply_moves_files(src, dest, rules):
    (src / "a.png").write_text("img")
    plan = build_plan(scan(src), rules, dest)
    moved, failures = apply_plan(plan)
    assert (moved, failures) == (1, [])
    assert (dest / "Images" / "a.png").read_text() == "img"
    assert not (src / "a.png").exists()


def test_apply_creates_missing_folders(src, dest, rules):
    (src / "a.png").write_text("img")
    apply_plan(build_plan(scan(src), rules, dest))
    assert (dest / "Images").is_dir()


def test_apply_reports_failures_without_stopping(src, dest, rules):
    (src / "a.png").write_text("one")
    (src / "b.pdf").write_text("two")
    plan = build_plan(scan(src), rules, dest)
    (src / "a.png").unlink()  # vanishes after planning, before applying
    moved, failures = apply_plan(plan)
    assert moved == 1
    assert len(failures) == 1


def test_apply_calls_the_progress_hook(src, dest, rules):
    (src / "a.png").write_text("img")
    seen = []
    apply_plan(build_plan(scan(src), rules, dest), on_move=seen.append)
    assert len(seen) == 1
