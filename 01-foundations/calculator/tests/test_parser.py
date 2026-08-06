import pytest

from calculator.ast_nodes import Assign, BinaryOp, Call, Number, UnaryOp, Variable
from calculator.parser import ParseError, parse


def test_single_number():
    assert parse("42") == Number(42.0)


def test_variable():
    assert parse("pi") == Variable("pi")


def test_addition_is_left_associative():
    # (1 + 2) + 3, not 1 + (2 + 3)
    tree = parse("1 + 2 + 3")
    assert tree == BinaryOp("+", BinaryOp("+", Number(1), Number(2)), Number(3))


def test_multiplication_binds_tighter_than_addition():
    assert parse("2 + 3 * 4") == BinaryOp("+", Number(2), BinaryOp("*", Number(3), Number(4)))


def test_parens_override_precedence():
    assert parse("(2 + 3) * 4") == BinaryOp("*", BinaryOp("+", Number(2), Number(3)), Number(4))


def test_unary_minus():
    assert parse("-5") == UnaryOp("-", Number(5))


def test_double_unary():
    assert parse("--5") == UnaryOp("-", UnaryOp("-", Number(5)))


def test_power_is_right_associative():
    # 2^(3^2) = 512, not (2^3)^2 = 64
    assert parse("2^3^2") == BinaryOp("^", Number(2), BinaryOp("^", Number(3), Number(2)))


def test_unary_minus_applies_after_power():
    # -2^2 is -(2^2), matching normal mathematical convention.
    assert parse("-2^2") == UnaryOp("-", BinaryOp("^", Number(2), Number(2)))


def test_negative_exponent_parses():
    assert parse("2^-1") == BinaryOp("^", Number(2), UnaryOp("-", Number(1)))


def test_star_star_is_the_same_operator_as_caret():
    assert parse("2**3") == parse("2^3")


def test_call_with_no_args():
    assert parse("rand()") == Call("rand", ())


def test_call_with_args():
    assert parse("max(1, 2 + 3)") == Call("max", (Number(1), BinaryOp("+", Number(2), Number(3))))


def test_nested_calls():
    assert parse("sqrt(abs(-4))") == Call("sqrt", (Call("abs", (UnaryOp("-", Number(4)),)),))


def test_assignment():
    assert parse("x = 5") == Assign("x", Number(5))


def test_assignment_is_right_associative():
    assert parse("x = y = 2") == Assign("x", Assign("y", Number(2)))


def test_assignment_captures_whole_expression():
    assert parse("x = 1 + 2") == Assign("x", BinaryOp("+", Number(1), Number(2)))


@pytest.mark.parametrize("src", ["", "1 +", "(1", "1)", "2 3", "max(1,)", ",", "* 2"])
def test_syntax_errors(src):
    with pytest.raises(ParseError):
        parse(src)


def test_error_message_includes_a_caret():
    with pytest.raises(ParseError) as exc:
        parse("1 + * 2")
    assert "^" in str(exc.value)
