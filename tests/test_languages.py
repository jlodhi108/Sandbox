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
