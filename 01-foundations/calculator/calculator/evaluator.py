"""Tree-walking evaluator.

Every failure a user can trigger -- unknown name, wrong argument count, divide
by zero, a root that leaves the reals -- is converted into ``EvalError`` with a
message worth reading. Nothing escapes as a bare Python traceback, because the
REPL and the GUI both surface these strings directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from calculator.ast_nodes import Assign, BinaryOp, Call, Node, Number, UnaryOp, Variable
from calculator.parser import parse


class EvalError(ValueError):
    """A well-formed expression that cannot produce a value."""


# --------------------------------------------------------------------- library

CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _factorial(x: float) -> float:
    if x < 0 or x != int(x):
        raise EvalError("factorial expects a non-negative whole number")
    if x > 170:
        raise EvalError("factorial overflows past 170")
    return float(math.factorial(int(x)))


def _log(x: float, base: float | None = None) -> float:
    return math.log(x) if base is None else math.log(x, base)


# name -> (function, min_args, max_args); max_args None means variadic.
FUNCTIONS: dict[str, tuple[Callable[..., float], int, int | None]] = {
    "sqrt": (math.sqrt, 1, 1),
    "cbrt": (lambda x: math.copysign(abs(x) ** (1 / 3), x), 1, 1),
    "abs": (abs, 1, 1),
    "sign": (lambda x: math.copysign(1.0, x) if x else 0.0, 1, 1),
    "exp": (math.exp, 1, 1),
    "log": (_log, 1, 2),
    "ln": (math.log, 1, 1),
    "log2": (math.log2, 1, 1),
    "log10": (math.log10, 1, 1),
    "sin": (math.sin, 1, 1),
    "cos": (math.cos, 1, 1),
    "tan": (math.tan, 1, 1),
    "asin": (math.asin, 1, 1),
    "acos": (math.acos, 1, 1),
    "atan": (math.atan, 1, 1),
    "atan2": (math.atan2, 2, 2),
    "sinh": (math.sinh, 1, 1),
    "cosh": (math.cosh, 1, 1),
    "tanh": (math.tanh, 1, 1),
    "deg": (math.degrees, 1, 1),
    "rad": (math.radians, 1, 1),
    "floor": (lambda x: float(math.floor(x)), 1, 1),
    "ceil": (lambda x: float(math.ceil(x)), 1, 1),
    "round": (lambda x, n=0: round(x, int(n)), 1, 2),
    "trunc": (lambda x: float(math.trunc(x)), 1, 1),
    "min": (min, 1, None),
    "max": (max, 1, None),
    "sum": (lambda *a: float(sum(a)), 1, None),
    "avg": (lambda *a: sum(a) / len(a), 1, None),
    "hypot": (math.hypot, 2, None),
    "fact": (_factorial, 1, 1),
    "gcd": (lambda a, b: float(math.gcd(int(a), int(b))), 2, 2),
}


@dataclass
class Environment:
    """Variable bindings. Constants are pre-seeded and may be shadowed."""

    variables: dict[str, float] = field(default_factory=lambda: dict(CONSTANTS))

    def get(self, name: str) -> float:
        if name in self.variables:
            return self.variables[name]
        hint = _suggest(name, list(self.variables) + list(FUNCTIONS))
        raise EvalError(f"unknown name {name!r}{hint}")

    def set(self, name: str, value: float) -> float:
        if name in FUNCTIONS:
            raise EvalError(f"{name!r} is a built-in function and cannot be reassigned")
        self.variables[name] = value
        return value


def _suggest(name: str, candidates: list[str]) -> str:
    """Cheap did-you-mean: same first letter and a near-identical length."""
    close = [
        c
        for c in candidates
        if c[:1] == name[:1] and abs(len(c) - len(name)) <= 1 and c != name
    ]
    return f" (did you mean {close[0]!r}?)" if close else ""


# ------------------------------------------------------------------ evaluation


def _binary(op: str, a: float, b: float) -> float:
    try:
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            if b == 0:
                raise EvalError("division by zero")
            return a / b
        if op == "//":
            if b == 0:
                raise EvalError("integer division by zero")
            return float(a // b)
        if op == "%":
            if b == 0:
                raise EvalError("modulo by zero")
            return a % b
        if op == "^":
            result = a**b
            if isinstance(result, complex):
                raise EvalError(f"{a} ^ {b} has no real result")
            return float(result)
    except OverflowError:
        raise EvalError("result too large to represent") from None
    except ZeroDivisionError:
        raise EvalError("division by zero") from None
    raise EvalError(f"unknown operator {op!r}")  # pragma: no cover


def evaluate_node(node: Node, env: Environment) -> float:
    if isinstance(node, Number):
        return node.value

    if isinstance(node, Variable):
        return env.get(node.name)

    if isinstance(node, UnaryOp):
        value = evaluate_node(node.operand, env)
        return -value if node.op == "-" else value

    if isinstance(node, BinaryOp):
        return _binary(node.op, evaluate_node(node.left, env), evaluate_node(node.right, env))

    if isinstance(node, Assign):
        return env.set(node.name, evaluate_node(node.value, env))

    if isinstance(node, Call):
        return _call(node, env)

    raise EvalError(f"cannot evaluate {type(node).__name__}")  # pragma: no cover


def _call(node: Call, env: Environment) -> float:
    entry = FUNCTIONS.get(node.name)
    if entry is None:
        hint = _suggest(node.name, list(FUNCTIONS))
        raise EvalError(f"unknown function {node.name!r}{hint}")

    fn, low, high = entry
    args = [evaluate_node(a, env) for a in node.args]

    if len(args) < low or (high is not None and len(args) > high):
        want = f"{low}" if low == high else f"{low}-{high}" if high else f"at least {low}"
        raise EvalError(f"{node.name}() takes {want} argument(s), got {len(args)}")

    try:
        return float(fn(*args))
    except EvalError:
        raise
    except ValueError as exc:
        raise EvalError(f"{node.name}(): {exc}") from None
    except OverflowError:
        raise EvalError(f"{node.name}(): result too large to represent") from None
    except ZeroDivisionError:
        raise EvalError(f"{node.name}(): division by zero") from None


def evaluate(source: str, env: Environment | None = None) -> float:
    """Parse and evaluate ``source``, returning a float."""
    return evaluate_node(parse(source), env or Environment())


def format_result(value: float) -> str:
    """Render a result the way a calculator display would."""
    if value != value:
        return "nan"
    if value in (math.inf, -math.inf):
        return "inf" if value > 0 else "-inf"
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    return f"{value:.10g}"


__all__ = ["evaluate", "evaluate_node", "Environment", "EvalError", "format_result", "FUNCTIONS"]
