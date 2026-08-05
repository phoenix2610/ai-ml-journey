"""A small expression calculator: tokenizer, parser, evaluator, two front-ends.

    >>> from calculator import evaluate
    >>> evaluate("2 + 3 * 4")
    14.0
"""

from calculator.tokens import tokenize, Token, TokenType, TokenizeError

__all__ = ["tokenize", "Token", "TokenType", "TokenizeError"]
__version__ = "0.1.0"
