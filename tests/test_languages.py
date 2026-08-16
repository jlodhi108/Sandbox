from languages import get_handler, get_handler_by_name
from languages.python_lang import PythonHandler
from languages.javascript_lang import JavaScriptHandler, TypeScriptHandler
from languages.java_lang import JavaHandler
from languages.php_lang import PhpHandler


def _assert_no_overlap(chunks):
    for i, a in enumerate(chunks):
        for b in chunks[i + 1:]:
            assert a.end_byte <= b.start_byte or b.end_byte <= a.start_byte


def test_get_handler_dispatches_by_extension():
    assert get_handler("foo.cpp").name == "cpp"
    assert get_handler("foo.py").name == "python"
    assert get_handler("foo.js").name == "javascript"
    assert get_handler("foo.ts").name == "typescript"
    assert get_handler("foo.java").name == "java"
    assert get_handler("foo.php").name == "php"


def test_get_handler_unknown_extension_raises():
    try:
        get_handler("foo.rb")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "rb" in str(e)


def test_get_handler_by_name_roundtrip():
    for name in ["cpp", "python", "javascript", "typescript", "java", "php"]:
        assert get_handler_by_name(name).name == name


def test_python_chunks_methods_and_functions_no_overlap():
    src = b"""
class Foo:
    def bar(self, x):
        return x * 2

def add(a, b):
    return a + b
"""
    chunks = PythonHandler().chunk(src)
    assert len(chunks) == 2
    _assert_no_overlap(chunks)


def test_javascript_chunks_function_and_method_no_overlap():
    src = b"""
function add(a, b) { return a + b; }
class Foo {
    bar(x) { return x * 2; }
}
"""
    chunks = JavaScriptHandler().chunk(src)
    assert len(chunks) == 2
    _assert_no_overlap(chunks)


def test_javascript_recognizes_const_arrow_function_as_valid_chunk():
    # A model modernizing a function_declaration into an arrow function is
    # the single most likely rewrite it'll make — the validator must
    # recognize this shape or it rejects every correct modernization.
    src = b"const greet = (name) => `Hello, ${name}!`;"
    chunks = JavaScriptHandler().chunk(src)
    assert len(chunks) == 1
    assert chunks[0].start_byte == 0
    assert chunks[0].end_byte == len(src)


def test_typescript_chunks_typed_function_and_method():
    src = b"""
function add(a: number, b: number): number { return a + b; }
class Foo {
    bar(x: number): number { return x * 2; }
}
"""
    chunks = TypeScriptHandler().chunk(src)
    assert len(chunks) == 2
    _assert_no_overlap(chunks)


def test_java_chunks_methods_only():
    src = b"""
public class Main {
    public static int add(int a, int b) { return a + b; }
    public void run() { System.out.println("hi"); }
}
"""
    chunks = JavaHandler().chunk(src)
    assert len(chunks) == 2
    _assert_no_overlap(chunks)


def test_php_chunks_function_and_method_no_overlap():
    src = b"""<?php
function add($a, $b) { return $a + $b; }
class Foo {
    public function bar($x) { return $x * 2; }
}
"""
    chunks = PhpHandler().chunk(src)
    assert len(chunks) == 2
    _assert_no_overlap(chunks)


def test_javascript_import_statement_uses_require_not_bare_import():
    # The base LanguageHandler default ("import {module}\n") isn't valid
    # JS/TS syntax at all — this asserts the override actually produces
    # something that parses, not just something different.
    stmt = JavaScriptHandler().import_statement("fs")
    assert stmt == "const fs = require('fs');\n"


def test_javascript_import_statement_sanitizes_unsafe_binding_names():
    # "fs/promises" and "node:fs" aren't valid JS identifiers as-is —
    # only the LOCAL BINDING needs sanitizing, the require() path keeps
    # the original string untouched.
    assert JavaScriptHandler().import_statement("fs/promises") == "const fs_promises = require('fs/promises');\n"
    assert JavaScriptHandler().import_statement("node:fs") == "const fs = require('node:fs');\n"


def test_javascript_has_import_detects_existing_require():
    source = "const fs = require('fs');\nfunction f() {}\n"
    assert JavaScriptHandler().has_import(source, "fs") is True
    assert JavaScriptHandler().has_import(source, "path") is False


def test_typescript_import_statement_includes_ambient_require_declaration():
    # Without @types/node in the sandbox, a bare require() fails to
    # typecheck ("Cannot find name 'require'") even though it's valid at
    # runtime — the ambient declaration is what makes tsc accept it.
    stmt = TypeScriptHandler().import_statement("fs")
    assert "declare function require(name: string): any;" in stmt
    assert "const fs = require('fs');" in stmt


def test_typescript_has_import_detects_existing_require():
    source = "declare function require(name: string): any;\nconst fs = require('fs');\n"
    assert TypeScriptHandler().has_import(source, "fs") is True
    assert TypeScriptHandler().has_import(source, "path") is False


def test_typescript_import_statement_skips_ambient_global_collisions():
    # console/crypto are ALSO ambient TS globals (from the default DOM
    # lib) — injecting `const crypto = require('crypto')` collides with
    # TS's own `declare var crypto` and fails to compile (TS2451:
    # "Cannot redeclare block-scoped variable"), confirmed by compiling
    # all 54 Node builtins through this project's exact tsc invocation.
    # Both are already-global at runtime too, so no import is correct.
    assert TypeScriptHandler().import_statement("crypto") == ""
    assert TypeScriptHandler().import_statement("console") == ""
    assert TypeScriptHandler().import_statement("node:crypto") == ""


def test_typescript_has_import_treats_ambient_globals_as_already_present():
    assert TypeScriptHandler().has_import("", "crypto") is True
    assert TypeScriptHandler().has_import("", "console") is True
