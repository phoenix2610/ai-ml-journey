import math

import pytest

from calculator.evaluator import Environment, EvalError, evaluate, format_result


@pytest.mark.parametrize(
    "src,expected",
    [
        ("1 + 1", 2),
        ("10 - 4", 6),
        ("6 * 7", 42),
        ("9 / 2", 4.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("2 ^ 10", 1024),
        ("2 ** 10", 1024),
    ],
)
def test_arithmetic(src, expected):
    assert evaluate(src) == expected


def test_precedence_and_associativity():
    assert evaluate("2 + 3 * 4") == 14
    assert evaluate("(2 + 3) * 4") == 20
    assert evaluate("2 ^ 3 ^ 2") == 512
    assert evaluate("-2 ^ 2") == -4
    assert evaluate("10 - 3 - 2") == 5


def test_unary():
    assert evaluate("-5") == -5
    assert evaluate("--5") == 5
    assert evaluate("+-3") == -3


def test_constants():
    assert evaluate("pi") == pytest.approx(math.pi)
    assert evaluate("e") == pytest.approx(math.e)
    assert evaluate("tau") == pytest.approx(math.tau)


@pytest.mark.parametrize(
    "src,expected",
    [
        ("sqrt(16)", 4),
        ("abs(-3)", 3),
        ("max(1, 5, 3)", 5),
        ("min(1, 5, 3)", 1),
        ("floor(3.7)", 3),
        ("ceil(3.2)", 4),
        ("round(3.14159, 2)", 3.14),
        ("log10(1000)", 3),
        ("log2(8)", 3),
        ("log(8, 2)", 3),
        ("fact(5)", 120),
        ("gcd(12, 18)", 6),
        ("hypot(3, 4)", 5),
        ("avg(2, 4, 6)", 4),
        ("sum(1, 2, 3)", 6),
        ("sign(-9)", -1),
        ("cbrt(-8)", -2),
    ],
)
def test_functions(src, expected):
    assert evaluate(src) == pytest.approx(expected)


def test_trig():
    assert evaluate("sin(0)") == pytest.approx(0)
    assert evaluate("cos(0)") == pytest.approx(1)
    assert evaluate("sin(pi / 2)") == pytest.approx(1)
    assert evaluate("deg(pi)") == pytest.approx(180)
    assert evaluate("rad(180)") == pytest.approx(math.pi)


def test_nested_calls():
    assert evaluate("sqrt(abs(-16))") == 4
    assert evaluate("max(sqrt(9), 2)") == 3


def test_variables_persist_in_an_environment():
    env = Environment()
    assert evaluate("x = 5", env) == 5
    assert evaluate("x * 2", env) == 10


def test_chained_assignment():
    env = Environment()
    evaluate("a = b = 3", env)
    assert evaluate("a + b", env) == 6


def test_assignment_can_reference_itself():
    env = Environment()
    evaluate("n = 10", env)
    evaluate("n = n + 5", env)
    assert evaluate("n", env) == 15


def test_constants_can_be_shadowed():
    env = Environment()
    evaluate("e = 1", env)
    assert evaluate("e", env) == 1


@pytest.mark.parametrize("src", ["1 / 0", "1 // 0", "1 % 0"])
def test_division_by_zero(src):
    with pytest.raises(EvalError, match="zero"):
        evaluate(src)


def test_unknown_variable():
    with pytest.raises(EvalError, match="unknown name"):
        evaluate("nope")


def test_unknown_function():
    with pytest.raises(EvalError, match="unknown function"):
        evaluate("nope(1)")


def test_did_you_mean_hint():
    with pytest.raises(EvalError, match="did you mean"):
        evaluate("sqr(4)")


def test_wrong_arity():
    with pytest.raises(EvalError, match="argument"):
        evaluate("sqrt(1, 2)")


def test_domain_error_is_wrapped():
    with pytest.raises(EvalError):
        evaluate("sqrt(-1)")


def test_cannot_reassign_a_builtin_function():
    with pytest.raises(EvalError, match="built-in"):
        evaluate("sqrt = 3")


def test_negative_base_fractional_exponent():
    with pytest.raises(EvalError, match="no real result"):
        evaluate("(-8) ^ 0.5")


@pytest.mark.parametrize(
    "value,text",
    [(2.0, "2"), (2.5, "2.5"), (-0.0, "0"), (1e20, "1e+20"), (math.inf, "inf")],
)
def test_format_result(value, text):
    assert format_result(value) == text
