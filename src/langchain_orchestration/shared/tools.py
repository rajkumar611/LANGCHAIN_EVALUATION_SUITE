"""Safe math evaluation and tool factories for LangChain endpoints."""

import ast
import operator as op
import re

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval_math(node: ast.expr) -> float:
    """Recursively evaluate an AST node using only whitelisted numeric operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_math(node.left), _safe_eval_math(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval_math(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def make_calculator_tool():
    """Return a LangChain calculator tool with safe AST-based math evaluation."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a maths expression, e.g. '25 * 4 + 10'."""
        if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\^]+", expression):
            return "Error: only numeric expressions are supported."
        try:
            tree = ast.parse(expression, mode="eval")
            return str(round(_safe_eval_math(tree.body), 10))
        except Exception as e:
            return f"Error: {e}"

    return calculator
