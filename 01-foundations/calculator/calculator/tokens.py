"""Lexical analysis: source text -> a flat stream of tokens.

Kept deliberately separate from the parser. The tokenizer's only job is to
decide *what* each character run is; deciding whether that sequence is legal
is the parser's problem. That split is what keeps both halves small.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    NUMBER = auto()
    IDENT = auto()
    OP = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    pos: int

    def __str__(self) -> str:
        if self.type is TokenType.EOF:
            return "end of input"
        return repr(self.value)


class TokenizeError(ValueError):
    """Raised on a character that cannot begin any token."""

    def __init__(self, message: str, pos: int, source: str) -> None:
        self.pos = pos
        self.source = source
        super().__init__(f"{message}\n{caret(source, pos)}")


def caret(source: str, pos: int) -> str:
    """Render the offending position under the source line, editor-style."""
    return f"  {source}\n  {' ' * pos}^"


# Longest first: '//' must win over '/', '**' over '*'.
OPERATORS = ("//", "**", "+", "-", "*", "/", "%", "^", "=")

NUMBER_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

SIMPLE = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    ",": TokenType.COMMA,
}


def tokenize(source: str) -> list[Token]:
    """Turn ``source`` into tokens, always terminated by a single EOF token."""
    tokens: list[Token] = []
    i = 0
    n = len(source)

    while i < n:
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        if ch in SIMPLE:
            tokens.append(Token(SIMPLE[ch], ch, i))
            i += 1
            continue

        # Numbers are checked before operators so that '.5' lexes as a number
        # rather than falling through to an unknown-character error.
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            m = NUMBER_RE.match(source, i)
            assert m is not None  # guarded by the branch condition
            tokens.append(Token(TokenType.NUMBER, m.group(), i))
            i = m.end()
            continue

        if ch.isalpha() or ch == "_":
            m = IDENT_RE.match(source, i)
            assert m is not None
            tokens.append(Token(TokenType.IDENT, m.group(), i))
            i = m.end()
            continue

        for op in OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token(TokenType.OP, op, i))
                i += len(op)
                break
        else:
            raise TokenizeError(f"unexpected character {ch!r}", i, source)

    tokens.append(Token(TokenType.EOF, "", n))
    return tokens
