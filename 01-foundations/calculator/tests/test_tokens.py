import pytest

from calculator.tokens import TokenizeError, TokenType, tokenize


def kinds(src):
    return [t.type for t in tokenize(src)[:-1]]


def values(src):
    return [t.value for t in tokenize(src)[:-1]]


def test_stream_always_ends_with_eof():
    assert tokenize("").pop().type is TokenType.EOF
    assert tokenize("1+1").pop().type is TokenType.EOF


def test_whitespace_is_not_significant():
    assert values("1+2") == values("  1   +  2 ") == ["1", "+", "2"]


@pytest.mark.parametrize(
    "src,expected",
    [
        ("42", "42"),
        ("3.14", "3.14"),
        ("2.", "2."),
        (".5", ".5"),
        ("1e9", "1e9"),
        ("1.5E-3", "1.5E-3"),
        ("6e+2", "6e+2"),
    ],
)
def test_number_forms(src, expected):
    assert values(src) == [expected]


def test_dot_before_digit_is_a_number_not_an_error():
    assert kinds(".5") == [TokenType.NUMBER]


@pytest.mark.parametrize("op", ["+", "-", "*", "/", "%", "^", "//", "**", "="])
def test_operators(op):
    assert values(f"1 {op} 2") == ["1", op, "2"]


def test_longest_operator_wins():
    # '//' must not lex as two '/' tokens.
    assert values("7//2") == ["7", "//", "2"]
    assert values("2**3") == ["2", "**", "3"]


def test_identifiers():
    assert values("pi + tau_2") == ["pi", "+", "tau_2"]
    assert kinds("sin") == [TokenType.IDENT]


def test_identifier_cannot_start_with_digit():
    # '2x' is a number followed by an identifier, which the parser will reject.
    assert values("2x") == ["2", "x"]


def test_grouping_and_commas():
    assert kinds("max(1, 2)") == [
        TokenType.IDENT,
        TokenType.LPAREN,
        TokenType.NUMBER,
        TokenType.COMMA,
        TokenType.NUMBER,
        TokenType.RPAREN,
    ]


def test_positions_are_recorded():
    tokens = tokenize("12 + 5")
    assert [t.pos for t in tokens[:-1]] == [0, 3, 5]


def test_unknown_character_reports_position():
    with pytest.raises(TokenizeError) as exc:
        tokenize("1 $ 2")
    assert exc.value.pos == 2
    assert "^" in str(exc.value)
