from unittest.mock import patch, MagicMock

from agents.nodes import (
    _strip_markdown_fence,
    _extract_required_imports,
    _validate_single_chunk,
    _invoke_llm_with_retry,
    _with_recipe,
    check_requires_resolvable,
    verifier_node,
    refactorer_node,
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


def test_check_requires_resolvable_accepts_python_stdlib():
    assert check_requires_resolvable("python", "pathlib") is True
    assert check_requires_resolvable("python", "collections") is True


def test_check_requires_resolvable_accepts_python_stdlib_dotted_submodule():
    # Only the top-level package name determines resolvability — a
    # dotted submodule path like "concurrent.futures" is still stdlib.
    assert check_requires_resolvable("python", "concurrent.futures") is True
    assert check_requires_resolvable("python", "urllib.request") is True


def test_check_requires_resolvable_rejects_python_third_party():
    # Real PyPI packages included — the sandbox never installs anything,
    # so these can never resolve regardless of being genuine packages.
    assert check_requires_resolvable("python", "numpy") is False
    assert check_requires_resolvable("python", "requests") is False
    assert check_requires_resolvable("python", "totally-hallucinated-pkg-xyz") is False


def test_check_requires_resolvable_accepts_node_builtins():
    assert check_requires_resolvable("javascript", "fs") is True
    assert check_requires_resolvable("typescript", "crypto") is True


def test_check_requires_resolvable_accepts_node_prefixed_builtins():
    assert check_requires_resolvable("javascript", "node:fs") is True


def test_check_requires_resolvable_rejects_npm_third_party():
    assert check_requires_resolvable("javascript", "lodash") is False
    assert check_requires_resolvable("typescript", "axios") is False


def test_check_requires_resolvable_returns_none_for_unchecked_languages():
    # C++/Java/PHP REQUIRES values (headers, FQCNs, namespaces) don't map
    # to a package-registry concept at all — a bad one fails fast and
    # clearly at the compile step instead, so this check opts out.
    assert check_requires_resolvable("cpp", "memory") is None
    assert check_requires_resolvable("java", "java.util.List") is None
    assert check_requires_resolvable("php", "Some\\Namespace\\ClassName") is None


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
        "original_code": "def main():\n    pass",
        "modernized_code": "def main():\n    pass",
        "required_imports": ["pathlib"],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
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


def test_verifier_node_fails_fast_on_hallucinated_package_without_calling_sandbox():
    state = {
        "language": "python",
        "full_source": b"def fetch(url):\n    pass\n",
        "chunk_start": 0,
        "chunk_end": 26,
        "modernized_code": "def fetch(url: str):\n    return requests.get(url)",
        "required_imports": ["requests"],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "requests" in result["compiler_stderr"]
        assert "third-party" in result["compiler_stderr"]
        mock_verify.assert_not_called()  # rejected before touching Docker at all


def test_validate_single_chunk_accepts_php_without_open_tag():
    # tree-sitter-php parses bare code (no <?php tag) as plain HTML/text
    # with zero function nodes — the validator must wrap it before
    # re-parsing, or every valid PHP response gets rejected as "0 chunks".
    handler = PhpHandler()
    code = "function greet(string $name): string {\n    return \"Hello, \" . $name . \"!\";\n}"
    assert _validate_single_chunk(handler, code) is None


def test_verifier_node_accepts_matching_baseline_stdout():
    state = {
        "language": "cpp",
        "full_source": b"int main() { return 0; }\n",
        "chunk_start": 0,
        "chunk_end": 25,
        "modernized_code": "int main() { return 0; }",
        "required_imports": [],
        "baseline_stdout": "same output\n",
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {
            "status": "success", "stderr": "", "stdout": "same output\n", "exit_code": 0,
        }
        result = verifier_node(state)
        assert result["status"] == "success"


def test_verifier_node_rejects_output_that_differs_from_baseline():
    # Reproduces the real failure class: something that compiles and exits
    # 0 but silently changed the program's behavior (e.g. the C++
    # .release() hack, or the duplicate-def corruption) — a bare
    # compile+run check would call this "success".
    state = {
        "language": "cpp",
        "full_source": b"int main() { return 0; }\n",
        "chunk_start": 0,
        "chunk_end": 25,
        "modernized_code": "int main() { return 0; }",
        "required_imports": [],
        "baseline_stdout": "expected output\n",
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {
            "status": "success", "stderr": "", "stdout": "DIFFERENT output\n", "exit_code": 0,
        }
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "changes the program's output" in result["compiler_stderr"]


def test_verifier_node_skips_equivalence_check_when_no_baseline():
    # baseline_stdout is None when the ORIGINAL file didn't run cleanly —
    # nothing to compare against, so a successful modernization should
    # still be accepted rather than blocked on an impossible check.
    state = {
        "language": "cpp",
        "full_source": b"int main() { return 0; }\n",
        "chunk_start": 0,
        "chunk_end": 25,
        "modernized_code": "int main() { return 0; }",
        "required_imports": [],
        "baseline_stdout": None,
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.return_value = {
            "status": "success", "stderr": "", "stdout": "whatever\n", "exit_code": 0,
        }
        result = verifier_node(state)
        assert result["status"] == "success"


def test_with_recipe_returns_base_prompt_unchanged_when_no_recipe():
    assert _with_recipe("BASE PROMPT", None) == "BASE PROMPT"
    assert _with_recipe("BASE PROMPT", "") == "BASE PROMPT"


def test_with_recipe_appends_instruction_to_base_prompt():
    result = _with_recipe("BASE PROMPT", "Only convert callbacks to async/await.")
    assert result.startswith("BASE PROMPT")
    assert "Only convert callbacks to async/await." in result


def test_refactorer_node_includes_recipe_instruction_in_system_prompt():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": "Only convert callback-style functions to async/await.",
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    system_message = mock_llm.invoke.call_args[0][0][0]
    assert "Only convert callback-style functions to async/await." in system_message.content


def test_refactorer_node_omits_recipe_section_when_none_configured():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": None,
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    system_message = mock_llm.invoke.call_args[0][0][0]
    assert "Additional guidance for this modernization run" not in system_message.content


def test_format_context_block_returns_empty_string_when_no_signatures():
    from agents.nodes import _format_context_block

    assert _format_context_block(None) == ""
    assert _format_context_block([]) == ""


def test_format_context_block_lists_signatures():
    from agents.nodes import _format_context_block

    block = _format_context_block(["def parse_config(path):", "class ConfigError(Exception):"])
    assert "def parse_config(path):" in block
    assert "class ConfigError(Exception):" in block
    assert "do not redefine" in block


def test_refactorer_node_includes_context_signatures_in_human_message():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": None,
        "context_signatures": ["def parse_config(path):", "def helper(y):"],
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "def parse_config(path):" in human_message.content
    assert "def helper(y):" in human_message.content
    assert "def f(x):" in human_message.content  # the actual chunk is still there


def test_refactorer_node_includes_context_signatures_in_fix_prompt_on_retry():
    state = {
        "language": "python",
        "iteration_count": 1,
        "original_code": "def f(x):\n    return x + 1\n",
        "modernized_code": "def f(x):\n    return x - 1\n",
        "compiler_stderr": "wrong output",
        "recipe_instruction": None,
        "context_signatures": ["def parse_config(path):"],
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes.escalation_llm", None):
        mock_llm.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "def parse_config(path):" in human_message.content


def test_refactorer_node_omits_context_block_when_no_signatures():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": None,
        "context_signatures": [],
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "already defined elsewhere in this codebase" not in human_message.content


def test_format_type_definitions_block_returns_empty_string_when_none():
    from agents.nodes import _format_type_definitions_block

    assert _format_type_definitions_block(None) == ""
    assert _format_type_definitions_block([]) == ""


def test_format_type_definitions_block_includes_full_definitions():
    from agents.nodes import _format_type_definitions_block

    block = _format_type_definitions_block(["class ConfigError(Exception):\n    pass"])
    assert "class ConfigError(Exception):" in block
    assert "do not redefine" in block


def test_refactorer_node_includes_type_definitions_in_human_message():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def load_config(path):\n    raise ConfigError('bad')\n",
        "recipe_instruction": None,
        "context_signatures": [],
        "referenced_type_definitions": ["class ConfigError(Exception):\n    pass"],
    }
    mock_response = MagicMock()
    mock_response.content = state["original_code"]
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "class ConfigError(Exception):" in human_message.content
    assert "def load_config(path):" in human_message.content


def test_refactorer_node_omits_type_definitions_block_when_none():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": None,
        "context_signatures": [],
        "referenced_type_definitions": [],
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "Full definitions of type" not in human_message.content


def test_format_exemplar_block_returns_empty_string_when_none():
    from agents.nodes import _format_exemplar_block

    assert _format_exemplar_block(None, None) == ""
    assert _format_exemplar_block("def f(a, b): return a+b", None) == ""
    assert _format_exemplar_block(None, "def f(a, b): return a+b") == ""


def test_format_exemplar_block_includes_before_and_after():
    from agents.nodes import _format_exemplar_block

    block = _format_exemplar_block("def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")
    assert "def f(a, b): return a+b" in block
    assert "def f(a: int, b: int) -> int: return a+b" in block
    assert "already successfully modernized" in block.lower()


def test_refactorer_node_includes_exemplar_on_first_attempt():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def g(x, y):\n    return x + y\n",
        "recipe_instruction": None,
        "context_signatures": [],
        "referenced_type_definitions": [],
        "exemplar_original": "def f(a, b): return a+b",
        "exemplar_modernized": "def f(a: int, b: int) -> int: return a+b",
    }
    mock_response = MagicMock()
    mock_response.content = state["original_code"]
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "def f(a, b): return a+b" in human_message.content
    assert "def f(a: int, b: int) -> int: return a+b" in human_message.content


def test_refactorer_node_omits_exemplar_on_retry():
    # Real compiler/behavioral error feedback on retry is more directly
    # useful than a generic worked example — the exemplar is first-
    # attempt-only by design.
    state = {
        "language": "python",
        "iteration_count": 1,
        "original_code": "def g(x, y):\n    return x + y\n",
        "modernized_code": "def g(x, y):\n    return x - y\n",
        "compiler_stderr": "wrong output",
        "recipe_instruction": None,
        "context_signatures": [],
        "referenced_type_definitions": [],
        "exemplar_original": "def f(a, b): return a+b",
        "exemplar_modernized": "def f(a: int, b: int) -> int: return a+b",
    }
    mock_response = MagicMock()
    mock_response.content = state["original_code"]
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes.escalation_llm", None):
        mock_llm.invoke.return_value = mock_response
        refactorer_node(state)

    human_message = mock_llm.invoke.call_args[0][0][1]
    assert "def f(a, b): return a+b" not in human_message.content


def test_refactorer_node_uses_deterministic_rule_without_calling_llm():
    state = {
        "language": "javascript",
        "iteration_count": 0,
        "original_code": "function f() {\n  var x = 1;\n  return x + 1;\n}",
        "recipe_instruction": None,
    }
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        result = refactorer_node(state)

    mock_llm.invoke.assert_not_called()
    mock_diversity.invoke.assert_not_called()
    assert result["used_deterministic_rule"] is True
    assert "const x = 1;" in result["modernized_code"]
    assert result["candidate_codes"] == [{"code": result["modernized_code"], "required_imports": []}]


def test_refactorer_node_falls_back_to_llm_when_no_deterministic_rule_applies():
    state = {
        "language": "python",
        "iteration_count": 0,
        "original_code": "def f(x):\n    return x + 1\n",
        "recipe_instruction": None,
    }
    mock_response = MagicMock()
    mock_response.content = "def f(x):\n    return x + 1\n"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        result = refactorer_node(state)

    mock_llm.invoke.assert_called()
    assert result["used_deterministic_rule"] is False


def test_refactorer_node_retry_after_failed_deterministic_candidate_uses_llm_and_marks_it():
    # iteration_count > 0 means the deterministic candidate from
    # iteration 0 already failed verification — the retry must go
    # through the normal LLM fix-prompt path AND correctly report
    # used_deterministic_rule=False for THIS (winning, LLM) attempt,
    # not leak True from the earlier failed deterministic attempt.
    state = {
        "language": "javascript",
        "iteration_count": 1,
        "original_code": "function f() {\n  var x = 1;\n  return x + 1;\n}",
        "modernized_code": "const x = 1;\nreturn x + 1;",
        "compiler_stderr": "some error",
        "recipe_instruction": None,
        "used_deterministic_rule": True,  # leftover from the failed iteration 0 attempt
    }
    mock_response = MagicMock()
    mock_response.content = "function f() {\n  const x = 1;\n  return x + 1;\n}"
    mock_response.usage_metadata = None
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes._diversity_llm") as mock_diversity:
        mock_llm.invoke.return_value = mock_response
        mock_diversity.invoke.return_value = mock_response
        result = refactorer_node(state)

    mock_llm.invoke.assert_called()
    assert result["used_deterministic_rule"] is False


def test_invoke_llm_with_retry_recovers_from_transient_failure():
    import httpx

    call_count = {"n": 0}

    def flaky_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise httpx.ConnectError("connection refused")
        return "ok response"

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = flaky_invoke
    with patch("agents.nodes.time.sleep") as mock_sleep:
        result = _invoke_llm_with_retry(mock_llm, [])
        assert result == "ok response"
        assert call_count["n"] == 2
        mock_sleep.assert_called_once()


def test_invoke_llm_with_retry_gives_up_after_max_attempts():
    import httpx

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = httpx.ConnectError("connection refused")
    with patch("agents.nodes.time.sleep"):
        try:
            _invoke_llm_with_retry(mock_llm, [])
            assert False, "expected ConnectionError"
        except ConnectionError as e:
            assert "ollama serve" in str(e)
        assert mock_llm.invoke.call_count == 3


def test_select_llm_uses_default_when_no_escalation_configured():
    import agents.nodes as nodes_module
    with patch.object(nodes_module, "escalation_llm", None):
        selected, used = nodes_module._select_llm(iteration_count=10)
        assert selected is nodes_module.llm
        assert used is False


def test_select_llm_stays_default_below_threshold():
    import agents.nodes as nodes_module
    fake_escalation = MagicMock()
    with patch.object(nodes_module, "escalation_llm", fake_escalation), \
         patch.object(nodes_module, "ESCALATION_THRESHOLD", 3):
        selected, used = nodes_module._select_llm(iteration_count=2)
        assert selected is nodes_module.llm
        assert used is False


def test_select_llm_escalates_at_threshold():
    import agents.nodes as nodes_module
    fake_escalation = MagicMock()
    with patch.object(nodes_module, "escalation_llm", fake_escalation), \
         patch.object(nodes_module, "ESCALATION_THRESHOLD", 3):
        selected, used = nodes_module._select_llm(iteration_count=3)
        assert selected is fake_escalation
        assert used is True


def test_assess_risk_parses_yes():
    from agents.nodes import assess_risk

    mock_response = MagicMock()
    mock_response.content = "RISK: yes\nThis function writes to a file."
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        risk_flag, reason = assess_risk("def save(x):\n    open('f').write(x)")
        assert risk_flag is True
        assert "writes to a file" in reason


def test_assess_risk_parses_no():
    from agents.nodes import assess_risk

    mock_response = MagicMock()
    mock_response.content = "RISK: no\nPure function, no side effects."
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        risk_flag, reason = assess_risk("def add(a, b):\n    return a + b")
        assert risk_flag is False
        assert "Pure function" in reason


def test_assess_risk_defaults_to_not_flagged_on_unparseable_response():
    # If the model doesn't follow the RISK: yes/no format, fail safe by
    # not blocking the pipeline on an unparseable response rather than
    # crashing — the structural/behavioral checks already did the real
    # verification work; this is a best-effort second opinion on top.
    from agents.nodes import assess_risk

    mock_response = MagicMock()
    mock_response.content = "I'm not sure, this seems fine I guess?"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        risk_flag, reason = assess_risk("def add(a, b):\n    return a + b")
        assert risk_flag is False


def test_assess_punt_parses_yes():
    from agents.nodes import assess_punt

    mock_response = MagicMock()
    mock_response.content = "PUNT: yes\nRelies on undocumented legacy quirks I'm unsure about."
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        punt_flag, reason = assess_punt("python", "def f(x):\n    return weird_legacy_thing(x)")
        assert punt_flag is True
        assert "legacy quirks" in reason


def test_assess_punt_parses_no():
    from agents.nodes import assess_punt

    mock_response = MagicMock()
    mock_response.content = "PUNT: no\nStraightforward, confident I can modernize this."
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        punt_flag, reason = assess_punt("python", "def add(a, b):\n    return a + b")
        assert punt_flag is False
        assert "Straightforward" in reason


def test_assess_punt_defaults_to_not_punted_on_unparseable_response():
    # Same fail-safe philosophy as assess_risk: an unparseable response
    # must not block the pipeline — default to attempting the chunk
    # normally rather than silently skipping it.
    from agents.nodes import assess_punt

    mock_response = MagicMock()
    mock_response.content = "unsure what to say here"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        punt_flag, _ = assess_punt("python", "def add(a, b):\n    return a + b")
        assert punt_flag is False


def test_assess_punt_uses_base_model_not_reviewer_or_escalation():
    # Deliberately the model's OWN self-assessment of whether IT can do
    # the job — unlike assess_risk, this should NOT route to a reviewer/
    # escalation model even if one is configured.
    from agents.nodes import assess_punt

    mock_response = MagicMock()
    mock_response.content = "PUNT: no\nfine"
    with patch("agents.nodes.llm") as mock_llm, \
         patch("agents.nodes.reviewer_llm") as mock_reviewer, \
         patch("agents.nodes.escalation_llm") as mock_escalation:
        mock_llm.invoke.return_value = mock_response
        assess_punt("python", "def add(a, b):\n    return a + b")
        mock_llm.invoke.assert_called_once()
        mock_reviewer.invoke.assert_not_called()
        mock_escalation.invoke.assert_not_called()


def test_select_reviewer_llm_prefers_explicit_reviewer_model():
    from agents.nodes import _select_reviewer_llm

    fake_reviewer = MagicMock()
    fake_escalation = MagicMock()
    with patch("agents.nodes.reviewer_llm", fake_reviewer), \
         patch("agents.nodes.escalation_llm", fake_escalation):
        assert _select_reviewer_llm(used_escalation=False) is fake_reviewer
        assert _select_reviewer_llm(used_escalation=True) is fake_reviewer


def test_select_reviewer_llm_uses_escalation_model_when_base_wrote_the_code():
    from agents.nodes import _select_reviewer_llm, llm

    fake_escalation = MagicMock()
    with patch("agents.nodes.reviewer_llm", None), \
         patch("agents.nodes.escalation_llm", fake_escalation):
        assert _select_reviewer_llm(used_escalation=False) is fake_escalation


def test_select_reviewer_llm_uses_base_model_when_escalation_wrote_the_code():
    from agents.nodes import _select_reviewer_llm, llm

    fake_escalation = MagicMock()
    with patch("agents.nodes.reviewer_llm", None), \
         patch("agents.nodes.escalation_llm", fake_escalation):
        assert _select_reviewer_llm(used_escalation=True) is llm


def test_select_reviewer_llm_falls_back_to_base_with_zero_extra_config():
    from agents.nodes import _select_reviewer_llm, llm

    with patch("agents.nodes.reviewer_llm", None), \
         patch("agents.nodes.escalation_llm", None):
        assert _select_reviewer_llm(used_escalation=False) is llm
        assert _select_reviewer_llm(used_escalation=True) is llm


def test_assess_risk_passes_used_escalation_through_to_reviewer_selection():
    from agents.nodes import assess_risk

    mock_response = MagicMock()
    mock_response.content = "RISK: no\nPure function."
    fake_escalation = MagicMock()
    fake_escalation.invoke.return_value = mock_response
    with patch("agents.nodes.reviewer_llm", None), \
         patch("agents.nodes.escalation_llm", fake_escalation), \
         patch("agents.nodes.llm") as mock_base_llm:
        risk_flag, _ = assess_risk("def add(a, b):\n    return a + b", used_escalation=False)
        assert risk_flag is False
        fake_escalation.invoke.assert_called_once()  # base wrote it, escalation reviews
        mock_base_llm.invoke.assert_not_called()


def test_generate_probes_returns_list_of_snippets():
    from agents.nodes import generate_probes

    mock_response = MagicMock()
    mock_response.content = "print(add(2, 3))\nprint(add(0, 0))\nprint(add(-1, 1))"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        probes = generate_probes("python", "def add(a, b):\n    return a + b", count=3)
        assert probes == ["print(add(2, 3))", "print(add(0, 0))", "print(add(-1, 1))"]


def test_generate_probes_returns_empty_list_on_skip():
    from agents.nodes import generate_probes

    mock_response = MagicMock()
    mock_response.content = "PROBE: SKIP"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        probes = generate_probes("python", "def connect(db_handle):\n    ...")
        assert probes == []


def test_generate_adversarial_probe_returns_snippet():
    from agents.nodes import generate_adversarial_probe

    mock_response = MagicMock()
    mock_response.content = "print(add(-1, 1))"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        snippet = generate_adversarial_probe(
            "python", "def add(a, b):\n    return a + b",
            "def add(a: int, b: int) -> int:\n    return a + b",
        )
        assert snippet == "print(add(-1, 1))"
        # Both versions must be in the prompt sent to the model — it
        # can't find a divergence between them without seeing both.
        sent_content = mock_llm.invoke.call_args[0][0][1].content
        assert "def add(a, b):" in sent_content
        assert "def add(a: int, b: int) -> int:" in sent_content


def test_generate_adversarial_probe_returns_none_on_skip():
    from agents.nodes import generate_adversarial_probe

    mock_response = MagicMock()
    mock_response.content = "PROBE: SKIP"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        snippet = generate_adversarial_probe("python", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b")
        assert snippet is None


def test_generate_adversarial_probe_takes_only_first_line():
    from agents.nodes import generate_adversarial_probe

    mock_response = MagicMock()
    mock_response.content = "print(add(-1, 1))\nsome extra commentary the model wasn't supposed to add"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        snippet = generate_adversarial_probe("python", "def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b")
        assert snippet == "print(add(-1, 1))"


def test_generate_adversarial_probe_returns_none_for_unsupported_language():
    from agents.nodes import generate_adversarial_probe

    with patch("agents.nodes.llm") as mock_llm:
        snippet = generate_adversarial_probe("cpp", "int add(int a, int b) { return a + b; }", "int add(int a, int b) { return a + b; }")
        assert snippet is None
        mock_llm.invoke.assert_not_called()


def test_wrap_call_as_probe_per_language():
    from agents.nodes import wrap_call_as_probe

    assert wrap_call_as_probe("python", "greet('Alice')") == "print(greet('Alice'))"
    assert wrap_call_as_probe("javascript", "greet('Alice')") == "console.log(greet('Alice'))"
    assert wrap_call_as_probe("php", "greet('Alice')") == "echo greet('Alice');"


def test_verifier_node_rejects_probe_output_mismatch():
    # The whole-file baseline wouldn't catch this at all if main() never
    # calls this function — the probe is the only thing that can.
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        # first call: the whole-file check (success, empty file has no output)
        # second call: the probe check (returns WRONG result) — determinism
        # re-check never fires since this probe already failed
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},
            {"status": "success", "stderr": "", "stdout": "99\n", "exit_code": 0},
        ]
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "different result" in result["compiler_stderr"]


def test_verifier_node_accepts_matching_probe_output():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
        # whole-file check, probe check, THEN the determinism re-check
        # (same probe run again) — all three must be consumed
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},
            {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},
            {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},
        ]
        result = verifier_node(state)
        assert result["status"] == "success"
        assert mock_verify.call_count == 3


def test_verifier_node_checks_all_probes_not_just_the_first():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [
            {"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"},
            {"snippet": "print(add(0, 0))", "baseline_stdout": "0\n"},
        ],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file
            {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},  # probe[0]
            {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},  # probe[0] determinism re-check
            {"status": "success", "stderr": "", "stdout": "WRONG\n", "exit_code": 0},  # probe[1] — mismatch
        ]
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "print(add(0, 0))" in result["compiler_stderr"]


def test_verifier_node_rejects_nondeterministic_output():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file
            {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},  # probe run 1 matches baseline
            {"status": "success", "stderr": "", "stdout": "7\n", "exit_code": 0},  # probe run 2 — DIFFERENT from run 1
        ]
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "non-deterministic" in result["compiler_stderr"]


def test_verifier_node_rejects_candidate_on_adversarial_divergence():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "modernized_code": "def add(a, b):\n    return a + b",  # pretend this is subtly wrong
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value="print(add(-1, 1))"):
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file check
            {"status": "success", "stderr": "", "stdout": "0\n", "exit_code": 0},  # adversarial probe vs ORIGINAL
            {"status": "success", "stderr": "", "stdout": "DIFFERENT\n", "exit_code": 0},  # vs CANDIDATE — diverges
        ]
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "adversarially-chosen input" in result["compiler_stderr"]
        assert "print(add(-1, 1))" in result["compiler_stderr"]


def test_verifier_node_accepts_candidate_when_adversarial_probe_agrees():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value="print(add(-1, 1))"):
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file check
            {"status": "success", "stderr": "", "stdout": "0\n", "exit_code": 0},  # vs ORIGINAL
            {"status": "success", "stderr": "", "stdout": "0\n", "exit_code": 0},  # vs CANDIDATE — matches
        ]
        result = verifier_node(state)
        assert result["status"] == "success"


def test_verifier_node_skips_adversarial_check_when_no_snippet_produced():
    # generate_adversarial_probe returning None (model couldn't/wouldn't
    # find a counterexample) must not block an otherwise-successful
    # candidate — same "no opinion" contract as every other probe path.
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None) as mock_gen:
        mock_verify.return_value = {"status": "success", "stderr": "", "stdout": "", "exit_code": 0}
        result = verifier_node(state)
        assert result["status"] == "success"
        mock_gen.assert_called_once()
        assert mock_verify.call_count == 1  # only the whole-file check — no probe round-trips


def test_verifier_node_skips_adversarial_check_when_it_cannot_run_against_original():
    # The model's counterexample failing to even RUN against the
    # ORIGINAL (bad guess, or references something out of scope) means
    # there's no baseline to compare against — skip, don't fail.
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value="print(add(undefined_name))"):
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file check
            {"status": "failed", "stderr": "NameError", "stdout": "", "exit_code": 1},  # vs ORIGINAL — doesn't even run
        ]
        result = verifier_node(state)
        assert result["status"] == "success"
        assert mock_verify.call_count == 2  # never reached the candidate-side check


def test_verifier_node_skips_adversarial_check_for_languages_without_function_probes():
    # cpp doesn't support function probes at all (see LanguageHandler.
    # supports_function_probe) — the adversarial check must respect the
    # same gate, never even asking the model for a counterexample.
    state = {
        "language": "cpp",
        "full_source": b"int add(int a, int b) { return a + b; }\n",
        "chunk_start": 0,
        "chunk_end": 40,
        "original_code": "int add(int a, int b) { return a + b; }",
        "modernized_code": "int add(int a, int b) { return a + b; }",
        "required_imports": [],
        "baseline_stdout": None,
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe") as mock_gen:
        mock_verify.return_value = {"status": "success", "stderr": "", "stdout": "", "exit_code": 0}
        result = verifier_node(state)
        assert result["status"] == "success"
        mock_gen.assert_not_called()


def test_verifier_node_rejects_candidate_on_property_test_failure():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a: int, b: int) -> int:\n    return a + b",
        "modernized_code": "def add(a: int, b: int) -> int:\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file check
            {"status": "failed", "stderr": "AssertionError: divergence with args {'a': 0, 'b': 0}", "stdout": "", "exit_code": 1},  # property test
        ]
        result = verifier_node(state)
        assert result["status"] == "failed"
        assert "Property-based testing" in result["compiler_stderr"]
        assert "divergence with args" in result["compiler_stderr"]


def test_verifier_node_accepts_candidate_when_property_test_passes():
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a: int, b: int) -> int:\n    return a + b",
        "modernized_code": "def add(a: int, b: int) -> int:\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
        mock_verify.side_effect = [
            {"status": "success", "stderr": "", "stdout": "", "exit_code": 0},   # whole-file check
            {"status": "success", "stderr": "", "stdout": "PROPERTY_TEST_OK\n", "exit_code": 0},  # property test
        ]
        result = verifier_node(state)
        assert result["status"] == "success"


def test_verifier_node_skips_property_test_when_no_type_hints():
    # Real code most of this project sees is UNANNOTATED legacy Python —
    # property_testing.generate_property_test returns None for it, and
    # that must not block an otherwise-successful candidate or trigger
    # an extra sandbox round-trip.
    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",  # no type hints
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
        mock_verify.return_value = {"status": "success", "stderr": "", "stdout": "", "exit_code": 0}
        result = verifier_node(state)
        assert result["status"] == "success"
        assert mock_verify.call_count == 1  # only the whole-file check


def test_verifier_node_skips_property_test_for_non_python_languages():
    state = {
        "language": "cpp",
        "full_source": b"int add(int a, int b) { return a + b; }\n",
        "chunk_start": 0,
        "chunk_end": 40,
        "original_code": "int add(int a, int b) { return a + b; }",
        "modernized_code": "int add(int a, int b) { return a + b; }",
        "required_imports": [],
        "baseline_stdout": None,
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.property_testing.generate_property_test") as mock_gen:
        mock_verify.return_value = {"status": "success", "stderr": "", "stdout": "", "exit_code": 0}
        result = verifier_node(state)
        assert result["status"] == "success"
        mock_gen.assert_not_called()


def test_scan_security_flags_when_findings_present():
    from agents.nodes import scan_security

    fake_findings = [{"rule_id": "sandbox.py-os-system", "line": 2, "message": "danger"}]
    with patch("agents.nodes.run_semgrep") as mock_semgrep:
        mock_semgrep.return_value = {"status": "success", "findings": fake_findings}
        flagged, findings = scan_security("python", "def f(cmd):\n    os.system(cmd)")
        assert flagged is True
        assert findings == fake_findings


def test_scan_security_not_flagged_when_clean():
    from agents.nodes import scan_security

    with patch("agents.nodes.run_semgrep") as mock_semgrep:
        mock_semgrep.return_value = {"status": "success", "findings": []}
        flagged, findings = scan_security("python", "def add(a, b):\n    return a + b")
        assert flagged is False
        assert findings == []


def test_scan_security_fails_safe_on_scanner_error():
    # If semgrep itself errors (timeout, bad output), must not block or
    # crash modernization — this is a best-effort extra signal on top of
    # checks that already proved the change is behaviorally safe.
    from agents.nodes import scan_security

    with patch("agents.nodes.run_semgrep") as mock_semgrep:
        mock_semgrep.return_value = {"status": "error", "findings": []}
        flagged, findings = scan_security("python", "def add(a, b):\n    return a + b")
        assert flagged is False
        assert findings == []


def test_scan_security_uses_correct_filename_for_language():
    # Real bug caught during development: semgrep picks its rule subset
    # by file EXTENSION. Scanning PHP code under "main.py" would
    # silently apply only Python rules and never find anything.
    from agents.nodes import scan_security

    with patch("agents.nodes.run_semgrep") as mock_semgrep:
        mock_semgrep.return_value = {"status": "success", "findings": []}
        scan_security("php", "<?php\nfunction f() {}")
        assert mock_semgrep.call_args[0][1] == "main.php"


def test_generate_mutant_strips_fence_and_extracts_requires():
    from agents.nodes import generate_mutant

    mock_response = MagicMock()
    mock_response.content = "```python\n# REQUIRES: math\ndef add(a, b):\n    return a - b\n```"
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        result = generate_mutant("python", "def add(a, b):\n    return a + b")
        assert result is not None
        mutant_code, required_imports = result
        assert mutant_code == "def add(a, b):\n    return a - b"
        assert required_imports == ["math"]


def test_generate_mutant_returns_none_when_identical_to_original():
    from agents.nodes import generate_mutant

    original = "def add(a, b):\n    return a + b"
    mock_response = MagicMock()
    mock_response.content = original
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        assert generate_mutant("python", original) is None


def test_generate_mutant_returns_none_on_empty_response():
    from agents.nodes import generate_mutant

    mock_response = MagicMock()
    mock_response.content = ""
    with patch("agents.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response
        assert generate_mutant("python", "def add(a, b):\n    return a + b") is None


def test_check_mutation_confidence_flags_when_mutant_passes_verification():
    # The mutant (a - b, deliberately wrong) is fed through the same
    # verification the real modernization already passed. If verify()
    # reports success for the mutant too, that means THIS chunk's checks
    # didn't distinguish broken from correct — the signal this whole
    # check exists to surface.
    from agents.nodes import check_mutation_confidence
    from languages.python_lang import PythonHandler

    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "original_code": "def add(a, b):\n    return a + b",
        "baseline_stdout": None,
        "probes": [],
    }
    mock_mutant_response = MagicMock()
    mock_mutant_response.content = "def add(a, b):\n    return a - b"
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes.verify") as mock_verify, \
         patch("agents.nodes.generate_adversarial_probe", return_value=None):
        mock_llm.invoke.return_value = mock_mutant_response
        mock_verify.return_value = {"status": "success", "stderr": "", "stdout": "", "exit_code": 0}
        flag, reason = check_mutation_confidence(
            PythonHandler(), state, "def add(a, b):\n    return a + b", []
        )
        assert flag is True
        assert "deliberately broken" in reason


def test_check_mutation_confidence_not_flagged_when_mutant_is_caught():
    # The mutant fails verification (as it should, being genuinely
    # broken) — this chunk's checks DO have real bite, no flag needed.
    from agents.nodes import check_mutation_confidence
    from languages.python_lang import PythonHandler

    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "baseline_stdout": None,
        "probes": [],
    }
    mock_mutant_response = MagicMock()
    mock_mutant_response.content = "def add(a, b):\n    return a - b"
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes.verify") as mock_verify:
        mock_llm.invoke.return_value = mock_mutant_response
        mock_verify.return_value = {"status": "failed", "stderr": "wrong output", "stdout": "", "exit_code": 1}
        flag, reason = check_mutation_confidence(
            PythonHandler(), state, "def add(a, b):\n    return a + b", []
        )
        assert flag is False
        assert reason == ""


def test_check_mutation_confidence_skips_cleanly_when_no_mutant_available():
    from agents.nodes import check_mutation_confidence
    from languages.python_lang import PythonHandler

    state = {
        "language": "python",
        "full_source": b"def add(a, b):\n    return a + b\n",
        "chunk_start": 0,
        "chunk_end": 32,
        "baseline_stdout": None,
        "probes": [],
    }
    mock_response = MagicMock()
    mock_response.content = "def add(a, b):\n    return a + b"  # identical -> no mutant
    with patch("agents.nodes.llm") as mock_llm, patch("agents.nodes.verify") as mock_verify:
        mock_llm.invoke.return_value = mock_response
        flag, reason = check_mutation_confidence(
            PythonHandler(), state, "def add(a, b):\n    return a + b", []
        )
        assert flag is False
        assert reason == ""
        mock_verify.assert_not_called()  # no mutant -> never even tries to verify one
