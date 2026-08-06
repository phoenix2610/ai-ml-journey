"""Recursive-descent parser: tokens -> syntax tree.

One function per precedence level, loosest binding at the top. The grammar the
code implements, in order:

    statement  := IDENT '=' statement | expression
    expression := term (('+' | '-') term)*
    term       := factor (('*' | '/' | '//' | '%') factor)*
    factor     := ('-' | '+') factor | power
    power      := primary (('^' | '**') factor)?
    primary    := NUMBER | IDENT | IDENT '(' args ')' | '(' expression ')'

Two deliberate asymmetries, both matching how people write maths on paper:

* ``power`` recurses into ``factor`` on its right, not into itself, so ``2^-1``
  parses and ``2^3^2`` is right-associative (512, not 64).
* ``factor`` sits *above* ``power``, so ``-2^2`` is ``-(2^2) == -4``.
"""

from __future__ import annotations

from calculator.ast_nodes import Assign, BinaryOp, Call, Node, Number, UnaryOp, Variable
from calculator.tokens import Token, TokenizeError, TokenType, caret, tokenize

ADDITIVE = {"+", "-"}
MULTIPLICATIVE = {"*", "/", "//", "%"}
POWER = {"^", "**"}


class ParseError(ValueError):
    def __init__(self, message: str, token: Token, source: str) -> None:
        self.token = token
        self.source = source
        super().__init__(f"{message}\n{caret(source, token.pos)}")


class Parser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.tokens = tokenize(source)
        self.i = 0

    # ---------------------------------------------------------------- helpers

    @property
    def current(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        token = self.tokens[self.i]
        if token.type is not TokenType.EOF:
            self.i += 1
        return token

    def at_op(self, ops: set[str]) -> bool:
        return self.current.type is TokenType.OP and self.current.value in ops

    def expect(self, type_: TokenType, what: str) -> Token:
        if self.current.type is not type_:
            raise ParseError(f"expected {what}, found {self.current}", self.current, self.source)
        return self.advance()

    # ---------------------------------------------------------------- grammar

    def parse(self) -> Node:
        if self.current.type is TokenType.EOF:
            raise ParseError("empty expression", self.current, self.source)
        node = self.statement()
        if self.current.type is not TokenType.EOF:
            raise ParseError(f"unexpected {self.current}", self.current, self.source)
        return node

    def statement(self) -> Node:
        # Assignment needs one token of lookahead: IDENT '=' but not IDENT '=='.
        if (
            self.current.type is TokenType.IDENT
            and self.tokens[self.i + 1].type is TokenType.OP
            and self.tokens[self.i + 1].value == "="
        ):
            name = self.advance().value
            self.advance()  # '='
            return Assign(name, self.statement())
        return self.expression()

    def expression(self) -> Node:
        node = self.term()
        while self.at_op(ADDITIVE):
            op = self.advance().value
            node = BinaryOp(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.factor()
        while self.at_op(MULTIPLICATIVE):
            op = self.advance().value
            node = BinaryOp(op, node, self.factor())
        return node

    def factor(self) -> Node:
        if self.at_op(ADDITIVE):
            op = self.advance().value
            return UnaryOp(op, self.factor())
        return self.power()

    def power(self) -> Node:
        base = self.primary()
        if self.at_op(POWER):
            self.advance()
            # Recurse into factor, not power: gives right-associativity and
            # allows a unary minus in the exponent.
            return BinaryOp("^", base, self.factor())
        return base

    def primary(self) -> Node:
        token = self.current

        if token.type is TokenType.NUMBER:
            self.advance()
            return Number(float(token.value))

        if token.type is TokenType.IDENT:
            self.advance()
            if self.current.type is TokenType.LPAREN:
                return Call(token.value, self.arguments())
            return Variable(token.value)

        if token.type is TokenType.LPAREN:
            self.advance()
            inner = self.expression()
            self.expect(TokenType.RPAREN, "')'")
            return inner

        raise ParseError(f"unexpected {token}", token, self.source)

    def arguments(self) -> tuple[Node, ...]:
        self.expect(TokenType.LPAREN, "'('")
        args: list[Node] = []
        if self.current.type is not TokenType.RPAREN:
            args.append(self.expression())
            while self.current.type is TokenType.COMMA:
                self.advance()
                args.append(self.expression())
        self.expect(TokenType.RPAREN, "')'")
        return tuple(args)


def parse(source: str) -> Node:
    """Parse ``source`` into a syntax tree."""
    return Parser(source).parse()


__all__ = ["parse", "Parser", "ParseError", "TokenizeError"]
