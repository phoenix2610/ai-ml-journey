"""Interactive prompt.

Keeps one Environment for the whole session, so variables and the automatic
``ans`` binding survive between lines. Meta-commands start with ':' to keep
them out of the expression namespace entirely.
"""

from __future__ import annotations

import sys

from calculator.evaluator import (
    FUNCTIONS,
    Environment,
    EvalError,
    evaluate,
    format_result,
)
from calculator.parser import ParseError
from calculator.tokens import TokenizeError

BANNER = """calc - expression calculator
type an expression, or :help for commands, :q to quit"""

HELP = """
  expressions   1 + 2 * 3      (2 + 3) ^ 2      -4 % 3      7 // 2
  variables     x = 5          x * 2            ans + 1
  functions     sqrt(2)        max(1, 2, 3)     log(8, 2)

  :help   this text          :vars   show variables
  :funcs  list functions     :clear  forget all variables
  :quit   exit  (also :q, Ctrl-D)
"""


def _try_readline() -> None:
    """Enable arrow-key history if readline is available. Optional everywhere."""
    try:
        import readline  # noqa: F401
    except ImportError:
        pass


def handle_command(cmd: str, env: Environment) -> bool:
    """Run a ':' command. Returns False when the session should end."""
    cmd = cmd.strip().lower()

    if cmd in (":q", ":quit", ":exit"):
        return False

    if cmd in (":h", ":help", ":?"):
        print(HELP)

    elif cmd == ":vars":
        user_vars = {k: v for k, v in env.variables.items() if k not in ("pi", "e", "tau", "inf")}
        if not user_vars:
            print("  no variables set")
        for name, value in sorted(user_vars.items()):
            print(f"  {name} = {format_result(value)}")

    elif cmd == ":funcs":
        names = sorted(FUNCTIONS)
        for i in range(0, len(names), 6):
            print("  " + "  ".join(f"{n:<9}" for n in names[i : i + 6]).rstrip())

    elif cmd == ":clear":
        env.variables = dict(Environment().variables)
        print("  variables cleared")

    else:
        print(f"  unknown command {cmd!r} -- try :help")

    return True


def run(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    # Non-interactive: `python -m calculator "2+2"` evaluates and exits.
    if argv:
        try:
            print(format_result(evaluate(" ".join(argv))))
            return 0
        except (EvalError, ParseError, TokenizeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    _try_readline()
    env = Environment()
    print(BANNER)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line.startswith(":"):
            if not handle_command(line, env):
                return 0
            continue

        try:
            result = evaluate(line, env)
        except (EvalError, ParseError, TokenizeError) as exc:
            print(f"  error: {exc}")
            continue

        env.variables["ans"] = result
        print(f"  {format_result(result)}")


__all__ = ["run", "handle_command"]
