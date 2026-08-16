"""Unit tests for agents.graph._capture_function_probes and its helpers —
the logic that combines real call sites (found via Tree-sitter) with
LLM-synthesized examples into the final probe list. Mocks generate_probes
and verify directly rather than going through the full graph; see
test_graph_integration.py for end-to-end graph wiring tests.
"""
from unittest.mock import patch

from agents.graph import _capture_function_probes, _find_real_call_site_probes
from languages.python_lang import PythonHandler

_OK = lambda stdout: {"status": "success", "stderr": "", "stdout": stdout, "exit_code": 0}


def test_find_real_call_site_probes_searches_own_file_and_siblings():
    handler = PythonHandler()
    full_source = b"def greet(name):\n    return name\n"
    siblings = [b"result = greet('Alice')\n", b"other_function(1)\n"]
    sites = _find_real_call_site_probes(handler, full_source, "def greet(name):\n    return name", siblings)
    assert sites == ["greet('Alice')"]


def test_capture_function_probes_dedupes_real_and_synthesized():
    full_source = b"def greet(name):\n    return name\n\nresult = greet('Alice')\n"
    original_code = "def greet(name):\n    return name"

    # The model independently generates the SAME example a real call
    # site already provides — must not appear twice in the final list.
    with patch("agents.graph.generate_probes") as mock_generate, \
         patch("agents.graph.verify") as mock_verify:
        mock_generate.return_value = ["print(greet('Alice'))", "print(greet(''))"]
        mock_verify.side_effect = [_OK("Alice\n"), _OK("Alice\n"), _OK("\n")]

        probes = _capture_function_probes("python", full_source, original_code, sibling_sources=[])

    snippets = [p["snippet"] for p in probes]
    assert snippets.count("print(greet('Alice'))") == 1
    assert "print(greet(''))" in snippets


def test_capture_function_probes_drops_probe_that_fails_against_original():
    full_source = b"def greet(name):\n    return name\n"
    original_code = "def greet(name):\n    return name"

    with patch("agents.graph.generate_probes") as mock_generate, \
         patch("agents.graph.verify") as mock_verify:
        mock_generate.return_value = ["print(greet('Alice'))", "print(greet(undefined_var))"]
        mock_verify.side_effect = [
            _OK("Alice\n"),
            {"status": "failed", "stderr": "NameError", "stdout": "", "exit_code": 1},
        ]
        probes = _capture_function_probes("python", full_source, original_code, sibling_sources=[])

    assert len(probes) == 1
    assert probes[0]["snippet"] == "print(greet('Alice'))"


def test_capture_function_probes_returns_empty_for_unsupported_language():
    probes = _capture_function_probes("java", b"...", "int add() {}", sibling_sources=[])
    assert probes == []
