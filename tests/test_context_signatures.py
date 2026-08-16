"""Unit tests for agents.graph._extract_context_signatures — the
sibling/codebase-signature grounding injected into the refactor prompt
(see agents/nodes.py:_format_context_block). Pure tree-sitter chunking,
no LLM/Docker involved.
"""
from agents.graph import _extract_context_signatures, MAX_CONTEXT_SIGNATURES, MAX_CONTEXT_SIBLING_FILES
from languages.python_lang import PythonHandler


def test_extracts_other_function_signatures_in_the_same_file():
    handler = PythonHandler()
    full_source = b"def parse_config(path):\n    return {}\n\ndef legacy_thing(x):\n    return x + 1\n"
    sigs = _extract_context_signatures(handler, full_source, 0, 0, None)
    assert sigs == ["def parse_config(path):", "def legacy_thing(x):"]


def test_excludes_the_chunk_being_modernized_itself():
    handler = PythonHandler()
    full_source = b"def a(x):\n    return x\n\ndef b(y):\n    return y\n"
    target = handler.chunk(full_source)[0]  # def a
    sigs = _extract_context_signatures(handler, full_source, target.start_byte, target.end_byte, None)
    assert sigs == ["def b(y):"]


def test_includes_a_bounded_sample_from_sibling_files():
    handler = PythonHandler()
    full_source = b"def a(x):\n    return x\n"
    siblings = [b"def c(z):\n    return z\n", b"def d(w):\n    return w\n"]
    sigs = _extract_context_signatures(handler, full_source, 0, 0, siblings)
    assert sigs == ["def a(x):", "def c(z):", "def d(w):"]


def test_deduplicates_identical_signatures_across_files():
    handler = PythonHandler()
    full_source = b"def a(x):\n    return x\n"
    siblings = [b"def a(x):\n    return x\n"]  # exact same signature, different file
    sigs = _extract_context_signatures(handler, full_source, 0, 0, siblings)
    assert sigs == ["def a(x):"]


def test_returns_empty_list_for_a_file_with_no_other_functions():
    handler = PythonHandler()
    full_source = b"def only_function(x):\n    return x\n"
    target = handler.chunk(full_source)[0]
    sigs = _extract_context_signatures(handler, full_source, target.start_byte, target.end_byte, None)
    assert sigs == []


def test_returns_empty_list_when_no_sibling_sources_given():
    handler = PythonHandler()
    full_source = b"def only_function(x):\n    return x\n"
    target = handler.chunk(full_source)[0]
    assert _extract_context_signatures(handler, full_source, target.start_byte, target.end_byte, []) == []


def test_caps_total_signatures_at_max_context_signatures():
    handler = PythonHandler()
    full_source = "\n\n".join(f"def f{i}(x):\n    return x" for i in range(MAX_CONTEXT_SIGNATURES + 10)).encode()
    sigs = _extract_context_signatures(handler, full_source, 0, 0, None)
    assert len(sigs) == MAX_CONTEXT_SIGNATURES


def test_caps_sibling_files_scanned_at_max_context_sibling_files():
    handler = PythonHandler()
    full_source = b"def a(x):\n    return x\n"
    # More sibling files than the cap, each contributing a UNIQUE signature —
    # only the first MAX_CONTEXT_SIBLING_FILES should ever be scanned.
    siblings = [f"def sib{i}(x):\n    return x\n".encode() for i in range(MAX_CONTEXT_SIBLING_FILES + 5)]
    sigs = _extract_context_signatures(handler, full_source, 0, 0, siblings)
    sibling_sigs = [s for s in sigs if s.startswith("def sib")]
    assert len(sibling_sigs) == MAX_CONTEXT_SIBLING_FILES


def test_gracefully_handles_a_sibling_file_in_a_different_language():
    # sibling_sources in repo mode is raw bytes of EVERY discovered file
    # regardless of language (see main.py's _read_all_file_contents) —
    # parsing non-Python content with the Python grammar must not crash,
    # same tolerance _find_real_call_site_probes already relies on.
    handler = PythonHandler()
    full_source = b"def a(x):\n    return x\n"
    js_sibling = b"function notPython(x) { return x + 1; }\n"
    sigs = _extract_context_signatures(handler, full_source, 0, 0, [js_sibling])
    assert "def a(x):" in sigs  # doesn't crash; garbage sibling content just contributes nothing useful
