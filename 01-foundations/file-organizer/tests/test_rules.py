from datetime import datetime
from pathlib import Path

import pytest

from organizer.rules import FileInfo, Rule, RuleError, RuleSet


def info(name, size=100, mtime=datetime(2024, 3, 15, 12, 0)):
    return FileInfo(path=Path("/src") / name, size=size, mtime=mtime)


# ------------------------------------------------------------------- FileInfo


def test_ext_is_lowercased_and_dotless():
    assert info("Photo.JPG").ext == "jpg"


def test_file_without_extension():
    assert info("README").ext == ""


def test_dotfile_has_no_extension():
    assert info(".bashrc").ext == ""


def test_template_vars():
    v = info("a.txt", mtime=datetime(2023, 7, 4)).template_vars()
    assert v["year"] == "2023"
    assert v["month"] == "07"
    assert v["month_name"] == "July"
    assert v["day"] == "04"


# ----------------------------------------------------------------- Rule.match


def test_extension_match():
    rule = Rule("img", "Images", extensions=frozenset({"png"}))
    assert rule.matches(info("a.png"))
    assert not rule.matches(info("a.txt"))


def test_extension_match_is_case_insensitive():
    rule = Rule("img", "Images", extensions=frozenset({"png"}))
    assert rule.matches(info("a.PNG"))


def test_glob_pattern_match():
    rule = Rule("shots", "Shots", patterns=("Screenshot*",))
    assert rule.matches(info("Screenshot 2024.png"))
    assert not rule.matches(info("photo.png"))


def test_size_bounds():
    rule = Rule("mid", "Mid", min_size=100, max_size=200)
    assert not rule.matches(info("a.bin", size=99))
    assert rule.matches(info("a.bin", size=150))
    assert not rule.matches(info("a.bin", size=201))


def test_size_only_rule_matches_anything_in_range():
    rule = Rule("big", "Big", min_size=1000)
    assert rule.matches(info("whatever.xyz", size=5000))


def test_size_gate_applies_even_when_extension_matches():
    rule = Rule("small_png", "Small", extensions=frozenset({"png"}), max_size=50)
    assert not rule.matches(info("a.png", size=999))


# --------------------------------------------------------------- destinations


def test_destination_templating():
    rule = Rule("img", "Images/{year}", extensions=frozenset({"png"}))
    assert rule.destination(info("a.png", mtime=datetime(2021, 1, 1))) == "Images/2021"


def test_destination_with_multiple_vars():
    rule = Rule("s", "Shots/{year}-{month}", patterns=("*",))
    assert rule.destination(info("a.png", mtime=datetime(2022, 9, 3))) == "Shots/2022-09"


def test_unknown_template_variable_raises():
    rule = Rule("bad", "X/{nope}", patterns=("*",))
    with pytest.raises(RuleError, match="template variable"):
        rule.destination(info("a.png"))


# -------------------------------------------------------------------- RuleSet


def test_highest_priority_wins():
    rs = RuleSet(
        rules=[
            Rule("images", "Images", extensions=frozenset({"png"})),
            Rule("shots", "Shots", patterns=("Screenshot*",), priority=20),
        ]
    )
    assert rs.categorise(info("Screenshot 1.png")) == "Shots"
    assert rs.categorise(info("holiday.png")) == "Images"


def test_tie_breaks_toward_the_first_declared_rule():
    rs = RuleSet(
        rules=[
            Rule("first", "First", extensions=frozenset({"dat"})),
            Rule("second", "Second", extensions=frozenset({"dat"})),
        ]
    )
    assert rs.categorise(info("a.dat")) == "First"


def test_unmatched_file_goes_to_fallback():
    rs = RuleSet(rules=[Rule("img", "Images", extensions=frozenset({"png"}))], fallback="Other")
    assert rs.categorise(info("mystery.qqq")) == "Other"


def test_negative_priority_loses_to_default():
    rs = RuleSet(
        rules=[
            Rule("huge", "Large", min_size=10, priority=-5),
            Rule("docs", "Documents", extensions=frozenset({"pdf"})),
        ]
    )
    assert rs.categorise(info("book.pdf", size=5000)) == "Documents"
    assert rs.categorise(info("blob.bin", size=5000)) == "Large"


# ---------------------------------------------------------------- TOML loading


def test_from_dict_round_trip():
    rs = RuleSet.from_dict(
        {
            "fallback": "Misc",
            "rule": [
                {"name": "img", "folder": "Images", "extensions": [".PNG", "jpg"], "priority": 3}
            ],
        }
    )
    assert rs.fallback == "Misc"
    assert rs.rules[0].extensions == frozenset({"png", "jpg"})
    assert rs.rules[0].priority == 3


def test_missing_required_key_raises():
    with pytest.raises(RuleError, match="folder"):
        RuleSet.from_dict({"rule": [{"name": "x"}]})


def test_missing_file_raises():
    with pytest.raises(RuleError, match="not found"):
        RuleSet.from_toml("/nonexistent/rules.toml")


def test_shipped_rules_file_is_valid():
    rules = Path(__file__).resolve().parent.parent / "rules.toml"
    rs = RuleSet.from_toml(rules)
    assert len(rs.rules) > 5
    assert rs.categorise(info("Screenshot 2024-01-01.png")) == "Images/Screenshots/2024-03"
    assert rs.categorise(info("report.pdf")) == "Documents"
    assert rs.categorise(info("data.csv")) == "Documents/Spreadsheets"
