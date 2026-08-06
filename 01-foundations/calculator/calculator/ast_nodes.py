"""Syntax tree node types.

Plain frozen dataclasses with no behaviour. Evaluation lives in evaluator.py so
that a second consumer -- a pretty-printer, a simplifier, a compiler -- can walk
the same tree without the nodes growing a method for each new use.
"""

from __future__ import annotations

from dataclasses import dataclass


class Node:
    """Base class for every syntax tree node."""

    __slots__ = ()


@dataclass(frozen=True)
class Number(Node):
    value: float

    def __repr__(self) -> str:
        return f"Number({self.value!r})"


@dataclass(frozen=True)
class Variable(Node):
    name: str

    def __repr__(self) -> str:
        return f"Variable({self.name!r})"


@dataclass(frozen=True)
class UnaryOp(Node):
    op: str
    operand: Node

    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r})"


@dataclass(frozen=True)
class BinaryOp(Node):
    op: str
    left: Node
    right: Node

    def __repr__(self) -> str:
        return f"BinaryOp({self.op!r}, {self.left!r}, {self.right!r})"


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: tuple[Node, ...]

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.args)
        return f"Call({self.name!r}, [{inner}])"


@dataclass(frozen=True)
class Assign(Node):
    name: str
    value: Node

    def __repr__(self) -> str:
        return f"Assign({self.name!r}, {self.value!r})"
