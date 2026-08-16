from unittest.mock import patch

from agents.nodes import (
    _strip_markdown_fence,
    _extract_required_imports,
    _validate_single_chunk,
    verifier_node,
)
from languages.python_lang import PythonHandler
from languages.php_lang import PhpHandler


def test_strip_markdown_fence_removes_cpp_fence():
    text = "```cpp\nint x = 1;\n```"
    assert _strip_markdown_fence(text) == "int x = 1;"


def test_strip_markdown_fence_noop_on_plain_code():
    text = "int x = 1;"
    assert _strip_markdown_fence(text) == "int x = 1;"


def test_extract_required_imports_finds_and_strips_marker_slash_style():
    text = "// REQUIRES: memory\nstd::unique_ptr<int> p;"
    clean, modules = _extract_required_imports(text)
    assert modules == ["memory"]
    assert "REQUIRES" not in clean
    assert clean == "std::unique_ptr<int> p;"


def test_extract_required_imports_finds_marker_hash_style():
    # Python-style comment prefix
    text = "# REQUIRES: pathlib\nPath('.').resolve()"
    clean, modules = _extract_required_imports(text)
    assert modules == ["pathlib"]
    assert "REQUIRES" not in clean


def test_extract_required_imports_multiple_markers():
    text = "// REQUIRES: memory\n// REQUIRES: vector\nstd::vector<std::unique_ptr<int>> v;"
    clean, modules = _extract_required_imports(text)
    assert modules == ["memory", "vector"]
    assert "REQUIRES" not in clean


def test_extract_required_imports_no_markers():
    text = "int add(int a, int b) { return a + b; }"
    clean, modules = _extract_required_imports(text)
    assert modules == []
    assert clean == text


def test_verifier_node_skips_import_already_present_cpp():
    state = {
        "language": "cpp",
        "full_source": b"#include <memory>\nint main() { return 0; }\n",
        "chunk_start": 18,
        "chunk_end": 45,
        "modernized_code": "int main() { return 0; }",
        "required_imports": ["memory"],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {"status": "success", "stderr": "", "exit_code": 0}
        verifier_node(state)
        candidate = mock_verify.call_args[0][0]
        # <memory> was already present, so we must not duplicate it
        assert candidate.count("#include <memory>") == 1


def test_verifier_node_injects_missing_import_cpp():
    state = {
        "language": "cpp",
        "full_source": b"int main() { return 0; }\n",
        "chunk_start": 0,
        "chunk_end": 25,
        "modernized_code": "int main() { return 0; }",
        "required_imports": ["memory"],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {"status": "success", "stderr": "", "exit_code": 0}
        verifier_node(state)
        candidate = mock_verify.call_args[0][0]
        assert "#include <memory>" in candidate


def test_verifier_node_injects_missing_import_python():
    state = {
        "language": "python",
        "full_source": b"def main():\n    pass\n",
        "chunk_start": 0,
        "chunk_end": 22,
        "modernized_code": "def main():\n    pass",
        "required_imports": ["pathlib"],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {"status": "success", "stderr": "", "exit_code": 0}
        verifier_node(state)
        candidate = mock_verify.call_args[0][0]
        assert "import pathlib" in candidate
        call_kwargs = mock_verify.call_args[1]
        assert call_kwargs["filename"] == "main.py"
        assert call_kwargs["run_cmd"] == "python3 main.py"


def test_validate_single_chunk_accepts_clean_single_function():
    handler = PythonHandler()
    code = "def greet(name: str) -> str:\n    return f\"Hello, {name}!\""
    assert _validate_single_chunk(handler, code) is None


def test_validate_single_chunk_rejects_duplicate_function():
    # Reproduces the real bug: the model appended a second, unrelated
    # function alongside the one it was asked to modernize.
    handler = PythonHandler()
    code = (
        "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n\n"
        "def read_config_path(base_dir, filename):\n    return base_dir + filename"
    )
    error = _validate_single_chunk(handler, code)
    assert error is not None
    assert "exactly ONE" in error


def test_validate_single_chunk_rejects_stray_literal_import():
    # Reproduces the real bug: the model wrote a real import instead of a
    # REQUIRES marker, corrupting what should be pure function output.
    handler = PythonHandler()
    code = "from pathlib import Path\n\ndef read_config_path(base_dir, filename):\n    return Path(base_dir) / filename"
    error = _validate_single_chunk(handler, code)
    assert error is not None
    assert "extra content" in error


def test_verifier_node_fails_fast_on_stray_duplicate_without_calling_sandbox():
    state = {
        "language": "python",
        "full_source": b"def greet(name):\n    return name\n",
        "chunk_start": 0,
        "chunk_end": 33,
        "modernized_code": (
            "def greet(name: str) -> str:\n    return name\n\n"
            "def extra(): pass"
        ),
        "required_imports": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "exactly ONE" in result["compiler_stderr"]
        mock_verify.assert_not_called()  # fails structurally before touching Docker


def test_validate_single_chunk_accepts_php_without_open_tag():
    # tree-sitter-php parses bare code (no <?php tag) as plain HTML/text
    # with zero function nodes — the validator must wrap it before
    # re-parsing, or every valid PHP response gets rejected as "0 chunks".
    handler = PhpHandler()
    code = "function greet(string $name): string {\n    return \"Hello, \" . $name . \"!\";\n}"
    assert _validate_single_chunk(handler, code) is None
