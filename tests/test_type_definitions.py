"""Unit tests for languages/base.py's extract_type_definitions and
agents.graph._extract_referenced_type_definitions — the "full type
definition" grounding layer on top of _extract_context_signatures'
one-line signatures.
"""
from agents.graph import _extract_referenced_type_definitions, MAX_TYPE_DEFINITIONS
from languages.python_lang import PythonHandler
from languages.javascript_lang import JavaScriptHandler, TypeScriptHandler
from languages.java_lang import JavaHandler
from languages.cpp import CppHandler
from languages.php_lang import PhpHandler


def test_extract_type_definitions_python():
    handler = PythonHandler()
    source = b"class Foo:\n    def bar(self):\n        pass\n"
    defs = handler.extract_type_definitions(source)
    assert set(defs.keys()) == {"Foo"}
    start, end = defs["Foo"]
    assert source[start:end] == b"class Foo:\n    def bar(self):\n        pass"


def test_extract_type_definitions_typescript_class_and_interface():
    handler = TypeScriptHandler()
    source = b"class Foo {}\ninterface Bar {}\n"
    defs = handler.extract_type_definitions(source)
    assert set(defs.keys()) == {"Foo", "Bar"}


def test_extract_type_definitions_java_class_and_interface():
    handler = JavaHandler()
    source = b"class Foo {}\ninterface Bar {}\n"
    defs = handler.extract_type_definitions(source)
    assert set(defs.keys()) == {"Foo", "Bar"}


def test_extract_type_definitions_cpp_class_and_struct():
    handler = CppHandler()
    source = b"class Foo {};\nstruct Bar {};\n"
    defs = handler.extract_type_definitions(source)
    assert set(defs.keys()) == {"Foo", "Bar"}


def test_extract_type_definitions_php_class_and_interface():
    handler = PhpHandler()
    source = b"<?php\nclass Foo {}\ninterface Bar {}\n"
    defs = handler.extract_type_definitions(source)
    assert set(defs.keys()) == {"Foo", "Bar"}


def test_extract_type_definitions_returns_empty_dict_when_no_query():
    class _NoQueryHandler(PythonHandler):
        type_definition_query_src = ""

    assert _NoQueryHandler().extract_type_definitions(b"class Foo:\n    pass\n") == {}


def test_referenced_type_definitions_includes_type_used_in_chunk():
    handler = PythonHandler()
    full_source = (
        b"class ConfigError(Exception):\n    def __init__(self, message):\n        self.message = message\n\n"
        b"def load_config(path):\n    if not path:\n        raise ConfigError('empty path')\n    return {}\n"
    )
    chunk_code = "def load_config(path):\n    if not path:\n        raise ConfigError('empty path')\n    return {}"
    defs = _extract_referenced_type_definitions(handler, chunk_code, full_source, None)
    assert len(defs) == 1
    assert "class ConfigError(Exception):" in defs[0]
    assert "self.message = message" in defs[0]


def test_referenced_type_definitions_excludes_unrelated_types():
    handler = PythonHandler()
    full_source = (
        b"class ConfigError(Exception):\n    pass\n\n"
        b"class UnrelatedThing:\n    pass\n\n"
        b"def load_config(path):\n    raise ConfigError('bad')\n"
    )
    chunk_code = "def load_config(path):\n    raise ConfigError('bad')"
    defs = _extract_referenced_type_definitions(handler, chunk_code, full_source, None)
    assert len(defs) == 1
    assert "ConfigError" in defs[0]
    assert "UnrelatedThing" not in "".join(defs)


def test_referenced_type_definitions_word_boundary_not_substring():
    # "Config" must NOT match "ConfigError" or vice versa — whole-word
    # matching only, not substring containment.
    handler = PythonHandler()
    full_source = b"class Config:\n    pass\n\ndef f(x):\n    return ConfigError(x)\n"
    chunk_code = "def f(x):\n    return ConfigError(x)"
    defs = _extract_referenced_type_definitions(handler, chunk_code, full_source, None)
    assert defs == []  # "Config" the class is never referenced, only "ConfigError"


def test_referenced_type_definitions_searches_sibling_files():
    handler = PythonHandler()
    full_source = b"def load_config(path):\n    raise ConfigError('bad')\n"
    sibling = b"class ConfigError(Exception):\n    pass\n"
    chunk_code = "def load_config(path):\n    raise ConfigError('bad')"
    defs = _extract_referenced_type_definitions(handler, chunk_code, full_source, [sibling])
    assert len(defs) == 1
    assert "ConfigError" in defs[0]


def test_referenced_type_definitions_returns_empty_for_language_without_query():
    handler = JavaScriptHandler()

    class _NoQueryJS(JavaScriptHandler):
        type_definition_query_src = ""

    defs = _extract_referenced_type_definitions(_NoQueryJS(), "function f() {}", b"function f() {}", None)
    assert defs == []


def test_referenced_type_definitions_caps_at_max_type_definitions():
    handler = PythonHandler()
    class_defs = "\n\n".join(f"class Type{i}:\n    pass" for i in range(MAX_TYPE_DEFINITIONS + 5))
    references = " ".join(f"Type{i}()" for i in range(MAX_TYPE_DEFINITIONS + 5))
    full_source = (class_defs + f"\n\ndef f():\n    return [{references}]").encode()
    chunk_code = f"def f():\n    return [{references}]"
    defs = _extract_referenced_type_definitions(handler, chunk_code, full_source, None)
    assert len(defs) == MAX_TYPE_DEFINITIONS


def test_referenced_type_definitions_uses_embeddings_ranking_when_enabled():
    from unittest.mock import patch

    handler = PythonHandler()
    full_source = b"def load_config(path):\n    raise ConfigError('bad')\n"
    sibling = b"class ConfigError(Exception):\n    pass\n"
    chunk_code = "def load_config(path):\n    raise ConfigError('bad')"

    with patch("agents.graph.embeddings.rank_by_relevance", return_value=[sibling]) as mock_rank:
        _extract_referenced_type_definitions(handler, chunk_code, full_source, [sibling])

    mock_rank.assert_called_once_with(chunk_code, [sibling])
