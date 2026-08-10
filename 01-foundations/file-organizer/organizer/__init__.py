"""Rule-driven file organizer with a dry-run planner and a reversible apply."""

from organizer.rules import FileInfo, Rule, RuleError, RuleSet

__all__ = ["FileInfo", "Rule", "RuleSet", "RuleError"]
__version__ = "0.1.0"
