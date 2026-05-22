"""Unit tests for shared utilities: safe math evaluator."""

import ast

import pytest

from src.langchain_orchestration.shared.tools import _safe_eval_math, make_calculator_tool


# Unit tests for _safe_eval_math (pure function, no HTTP calls)
def test_safe_eval_math_addition():
    """Test safe math evaluator handles addition."""
    tree = ast.parse("2 + 3", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 5


def test_safe_eval_math_subtraction():
    """Test safe math evaluator handles subtraction."""
    tree = ast.parse("10 - 3", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 7


def test_safe_eval_math_multiplication():
    """Test safe math evaluator handles multiplication."""
    tree = ast.parse("4 * 5", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 20


def test_safe_eval_math_division():
    """Test safe math evaluator handles division."""
    tree = ast.parse("20 / 4", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 5.0


def test_safe_eval_math_power():
    """Test safe math evaluator handles exponentiation."""
    tree = ast.parse("2 ** 3", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 8


def test_safe_eval_math_unary_neg():
    """Test safe math evaluator handles unary negation."""
    tree = ast.parse("-5", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == -5


def test_safe_eval_math_complex_expression():
    """Test safe math evaluator handles complex expression."""
    tree = ast.parse("2 + 3 * 4", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 14


def test_safe_eval_math_float():
    """Test safe math evaluator handles floats."""
    tree = ast.parse("1.5 + 2.5", mode="eval")
    result = _safe_eval_math(tree.body)
    assert result == 4.0


def test_safe_eval_math_invalid_operation():
    """Test safe math evaluator rejects unsupported operations."""
    tree = ast.parse("5 | 3", mode="eval")  # bitwise OR
    with pytest.raises(ValueError):
        _safe_eval_math(tree.body)


def test_safe_eval_math_invalid_string():
    """Test safe math evaluator rejects string constants."""
    tree = ast.parse('"hello"', mode="eval")
    with pytest.raises(ValueError):
        _safe_eval_math(tree.body)
