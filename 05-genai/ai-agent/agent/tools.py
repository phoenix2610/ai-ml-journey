"""Tools the agent can call, and schemas generated from Python signatures.

The schema is derived from the function's type hints and docstring rather than
written by hand. Hand-maintained schemas drift from the code they describe, and
the failure is silent: the model calls a tool with arguments that were valid
last month, gets a TypeError, and burns a step recovering.

Every tool returns a `ToolResult` rather than raising. An agent needs to *see*
that a call failed so it can adapt -- an exception propagating out of the loop
ends the run, which is exactly the wrong response to "file not found".
"""

from __future__ import annotations

import ast
import inspect
import json
import operator
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints

JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class ToolResult:
    ok: bool
    value: Any = None
    error: str = ""
    duration_ms: float = 0.0

    def __str__(self) -> str:
        return str(self.value) if self.ok else f"ERROR: {self.error}"

    def for_model(self) -> Any:
        """What gets sent back as the functionResponse payload."""
        return self.value if self.ok else {"error": self.error}


@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: dict
    dangerous: bool = False

    def schema(self) -> dict:
        """Gemini functionDeclaration form."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __call__(self, **kwargs) -> ToolResult:
        started = time.monotonic()
        try:
            value = self.function(**kwargs)
        except Exception as exc:
            # Deliberately broad: a tool raising must become an observation the
            # agent can react to, never an exception that kills the run.
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - started) * 1000,
            )
        return ToolResult(
            ok=True, value=value, duration_ms=(time.monotonic() - started) * 1000
        )


def build_parameters(function: Callable) -> dict:
    """JSON Schema from a signature. Required = parameters with no default."""
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    doc = inspect.getdoc(function) or ""
    param_docs = _parse_param_docs(doc)

    for name, parameter in signature.parameters.items():
        if name in ("self", "cls") or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(name, str)
        properties[name] = {
            "type": JSON_TYPES.get(annotation, "string"),
            "description": param_docs.get(name, ""),
        }
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _parse_param_docs(doc: str) -> dict[str, str]:
    """Pull `name: description` lines out of an Args: block."""
    out: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if not stripped or stripped.endswith(":"):
                break
            if ":" in stripped:
                name, _, description = stripped.partition(":")
                out[name.strip()] = description.strip()
    return out


def tool(function: Callable | None = None, *, dangerous: bool = False) -> Any:
    """Turn a plain function into a Tool. Usable bare or with arguments."""

    def wrap(f: Callable) -> Tool:
        doc = inspect.getdoc(f) or f.__name__
        summary = doc.split("\n\n")[0].split("Args:")[0].strip()
        return Tool(
            name=f.__name__,
            description=summary,
            function=f,
            parameters=build_parameters(f),
            dangerous=dangerous,
        )

    return wrap(function) if function else wrap


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ValueError(f"tool {t.name!r} is already registered")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> ToolResult:
        t = self._tools.get(name)
        if t is None:
            # A hallucinated tool name is an observation, not a crash.
            return ToolResult(
                ok=False,
                error=f"no such tool {name!r}. Available: {', '.join(self.names())}",
            )

        allowed = set(t.parameters.get("properties", {}))
        unknown = set(args) - allowed
        if unknown:
            return ToolResult(
                ok=False,
                error=f"{name} got unexpected argument(s) {sorted(unknown)}; "
                      f"accepts {sorted(allowed)}",
            )

        missing = set(t.parameters.get("required", [])) - set(args)
        if missing:
            return ToolResult(ok=False, error=f"{name} is missing {sorted(missing)}")

        return t(**args)

    def __len__(self) -> int:
        return len(self._tools)

    def __bool__(self) -> bool:
        return True


# --------------------------------------------------------------- the tools

SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression and return the number.

    Args:
        expression: Arithmetic only, e.g. "(13772 * 3) / 2". No names or calls.
    """
    # Parsed and walked rather than eval'd: eval on model-authored strings is
    # arbitrary code execution by a system whose whole job is authoring strings.
    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.BinOp):
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator {type(node.op).__name__}")
            return op(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported unary operator")
            return op(evaluate(node.operand))
        raise ValueError(f"unsupported syntax {type(node).__name__}")

    return float(evaluate(ast.parse(expression, mode="eval").body))


@tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents.

    Args:
        path: Path to the file.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if resolved.is_dir():
        raise IsADirectoryError(f"{path} is a directory; use list_files")
    text = resolved.read_text(encoding="utf-8", errors="replace")
    # Truncate: a 5 MB file would blow the context window and the budget.
    return text if len(text) <= 20_000 else text[:20_000] + "\n... [truncated]"


@tool
def list_files(directory: str = ".", pattern: str = "*") -> list:
    """List files in a directory, optionally filtered by a glob pattern.

    Args:
        directory: Directory to list.
        pattern: Glob such as "*.py".
    """
    base = Path(directory).expanduser()
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    return sorted(str(p.relative_to(base)) for p in base.glob(pattern) if p.is_file())[:200]


@tool(dangerous=True)
def write_file(path: str, content: str) -> str:
    """Write text to a file, creating parent directories. Overwrites.

    Args:
        path: Destination path.
        content: Text to write.
    """
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {resolved}"


@tool
def search_text(directory: str, query: str, pattern: str = "*.md") -> list:
    """Find files containing a string, with matching line numbers.

    Args:
        directory: Directory to search.
        query: Case-insensitive substring.
        pattern: Glob limiting which files are searched.
    """
    base = Path(directory).expanduser()
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    needle = query.lower()
    matches = []
    for path in sorted(base.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if needle in line.lower():
                    matches.append(f"{path}:{number}: {line.strip()[:160]}")
                    if len(matches) >= 50:
                        return matches
        except OSError:
            continue
    return matches


@tool
def finish(answer: str) -> str:
    """Signal that the task is complete and give the final answer.

    Args:
        answer: The complete answer for the user.
    """
    return answer


def default_registry(*, allow_writes: bool = False) -> ToolRegistry:
    """The standard toolset. Writing is opt-in."""
    tools = [calculate, read_file, list_files, search_text, finish]
    if allow_writes:
        tools.append(write_file)
    return ToolRegistry(tools)


__all__ = [
    "Tool", "ToolResult", "ToolRegistry", "tool", "build_parameters",
    "calculate", "read_file", "list_files", "write_file", "search_text", "finish",
    "default_registry",
]
