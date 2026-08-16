from unittest.mock import patch, MagicMock

from agents.nodes import (
    _strip_markdown_fence,
    _extract_required_imports,
    _validate_single_chunk,
    _invoke_llm_with_retry,
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
        "modernized_code": "def add(a, b):\n    return a + b",
        "required_imports": [],
        "baseline_stdout": None,
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
    }
    with patch("agents.nodes.verify") as mock_verify:
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
