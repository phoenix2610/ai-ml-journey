import pytest

from agent.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    build_parameters,
    calculate,
    default_registry,
    finish,
    list_files,
    read_file,
    search_text,
    tool,
    write_file,
)


# ------------------------------------------------------------ schema building


def test_schema_is_derived_from_the_signature():
    @tool
    def sample(name: str, count: int, ratio: float = 1.0) -> str:
        """Do a thing.

        Args:
            name: What to call it.
            count: How many.
            ratio: Optional scaling.
        """
        return name

    schema = sample.schema()
    properties = schema["parameters"]["properties"]
    assert properties["name"]["type"] == "string"
    assert properties["count"]["type"] == "integer"
    assert properties["ratio"]["type"] == "number"


def test_required_is_parameters_without_defaults():
    @tool
    def sample(needed: str, optional: str = "x") -> str:
        """Summary."""
        return needed

    assert sample.parameters["required"] == ["needed"]


def test_parameter_descriptions_come_from_the_docstring():
    @tool
    def sample(path: str) -> str:
        """Read something.

        Args:
            path: Where the file lives.
        """
        return path

    assert sample.parameters["properties"]["path"]["description"] == "Where the file lives."


def test_description_is_the_docstring_summary():
    assert calculate.description.startswith("Evaluate an arithmetic expression")
    assert "Args:" not in calculate.description


def test_a_tool_with_no_parameters_has_no_required_key():
    @tool
    def sample() -> str:
        """Nothing needed."""
        return "ok"

    assert "required" not in sample.parameters


def test_unannotated_parameters_default_to_string():
    @tool
    def sample(thing) -> str:
        """Summary."""
        return str(thing)

    assert sample.parameters["properties"]["thing"]["type"] == "string"


def test_dangerous_flag_is_carried():
    assert write_file.dangerous is True
    assert read_file.dangerous is False


# ------------------------------------------------------------------- results


def test_success_returns_a_value():
    result = calculate(expression="2 + 3")
    assert result.ok and result.value == 5.0


def test_failure_is_returned_not_raised():
    """The agent must be able to see and react to a failed call."""
    result = read_file(path="/definitely/not/here.txt")
    assert result.ok is False
    assert "FileNotFoundError" in result.error


def test_result_stringifies_usefully():
    assert "ERROR" in str(read_file(path="/nope"))
    assert str(calculate(expression="1+1")) == "2.0"


def test_for_model_wraps_errors():
    assert "error" in read_file(path="/nope").for_model()


def test_duration_is_recorded():
    assert calculate(expression="1+1").duration_ms >= 0


# ----------------------------------------------------------------- calculate


@pytest.mark.parametrize(
    "expression,expected",
    [("2+3", 5), ("10 / 4", 2.5), ("2 ** 8", 256), ("-5 + 2", -3), ("(1+2)*3", 9), ("7 // 2", 3)],
)
def test_arithmetic(expression, expected):
    assert calculate(expression=expression).value == expected


def test_calculate_refuses_names():
    """No eval: a model-authored string must not become executable code."""
    result = calculate(expression="__import__('os').system('echo hi')")
    assert result.ok is False


def test_calculate_refuses_function_calls():
    assert calculate(expression="open('/etc/passwd')").ok is False


def test_calculate_rejects_nonsense():
    assert calculate(expression="not an expression").ok is False


def test_calculate_reports_division_by_zero():
    result = calculate(expression="1/0")
    assert result.ok is False and "ZeroDivision" in result.error


# --------------------------------------------------------------- file tools


def test_read_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("hello")
    assert read_file(path=str(path)).value == "hello"


def test_read_directory_is_an_error(tmp_path):
    assert read_file(path=str(tmp_path)).ok is False


def test_read_file_truncates_huge_files(tmp_path):
    path = tmp_path / "big.txt"
    path.write_text("x" * 50_000)
    assert "truncated" in read_file(path=str(path)).value


def test_list_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")
    assert set(list_files(directory=str(tmp_path)).value) == {"a.py", "b.md"}


def test_list_files_with_a_pattern(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")
    assert list_files(directory=str(tmp_path), pattern="*.py").value == ["a.py"]


def test_list_missing_directory_is_an_error():
    assert list_files(directory="/nope/nothing").ok is False


def test_write_file(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    assert write_file(path=str(target), content="data").ok
    assert target.read_text() == "data"


def test_search_text(tmp_path):
    (tmp_path / "a.md").write_text("nothing\nthe threshold was 0.1141\n")
    (tmp_path / "b.md").write_text("unrelated")
    matches = search_text(directory=str(tmp_path), query="0.1141").value
    assert len(matches) == 1 and "a.md:2" in matches[0]


def test_search_is_case_insensitive(tmp_path):
    (tmp_path / "a.md").write_text("The Threshold")
    assert search_text(directory=str(tmp_path), query="threshold").value


def test_finish_returns_the_answer():
    assert finish(answer="all done").value == "all done"


# ------------------------------------------------------------------ registry


@pytest.fixture
def registry():
    return default_registry()


def test_default_registry_excludes_writes(registry):
    assert "write_file" not in registry.names()
    assert "read_file" in registry.names()


def test_writes_can_be_enabled():
    assert "write_file" in default_registry(allow_writes=True).names()


def test_schemas_are_produced_for_every_tool(registry):
    schemas = registry.schemas()
    assert len(schemas) == len(registry)
    assert all({"name", "description", "parameters"} <= set(s) for s in schemas)


def test_call_dispatches(registry):
    assert registry.call("calculate", {"expression": "6*7"}).value == 42


def test_unknown_tool_is_an_observation_not_a_crash(registry):
    """A hallucinated tool name must not end the run."""
    result = registry.call("summon_unicorn", {"n": 1})
    assert result.ok is False
    assert "no such tool" in result.error
    assert "calculate" in result.error          # tells the model what exists


def test_unexpected_argument_is_reported(registry):
    result = registry.call("calculate", {"expression": "1+1", "colour": "blue"})
    assert result.ok is False and "unexpected argument" in result.error


def test_missing_argument_is_reported(registry):
    result = registry.call("calculate", {})
    assert result.ok is False and "missing" in result.error


def test_duplicate_registration_raises():
    r = ToolRegistry([calculate])
    with pytest.raises(ValueError, match="already registered"):
        r.register(calculate)


def test_empty_registry_is_truthy():
    """__len__ without __bool__ would make an empty registry falsy."""
    assert bool(ToolRegistry([])) is True
    assert len(ToolRegistry([])) == 0
