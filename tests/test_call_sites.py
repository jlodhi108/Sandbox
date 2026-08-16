from languages.python_lang import PythonHandler
from languages.javascript_lang import JavaScriptHandler, TypeScriptHandler
from languages.php_lang import PhpHandler
from languages.java_lang import JavaHandler


def test_python_extract_function_name():
    h = PythonHandler()
    assert h.extract_function_name("def greet(name):\n    return name") == "greet"


def test_python_find_call_sites_excludes_method_calls():
    h = PythonHandler()
    source = b"greet('Alice')\nresult = add(2, 3)\nobj.greet(1)\n"
    sites = h.find_call_sites("greet", source)
    assert sites == ["greet('Alice')"]


def test_python_find_call_sites_multiple_matches():
    h = PythonHandler()
    source = b"greet('Alice')\nprint(greet('Bob'))\n"
    sites = h.find_call_sites("greet", source)
    assert sites == ["greet('Alice')", "greet('Bob')"]


def test_python_find_call_sites_no_matches():
    h = PythonHandler()
    source = b"other_function(1, 2)\n"
    assert h.find_call_sites("greet", source) == []


def test_python_find_call_sites_handles_nested_calls_correctly():
    # Regression: a naive "does any @fname fall inside this @call's byte
    # range" check wrongly matches print(greet('Bob')) as a call to
    # "greet" for the OUTER print() call too, since greet's nested call
    # sits entirely within print(...)'s range. Only the inner greet(...)
    # call should be reported.
    h = PythonHandler()
    source = b"print(greet('Bob'))\n"
    assert h.find_call_sites("greet", source) == ["greet('Bob')"]
    # print() is itself a real call node too — must be found on its own
    # terms, not accidentally merged with or excluded because of the
    # nested greet() call inside its arguments.
    assert h.find_call_sites("print", source) == ["print(greet('Bob'))"]


def test_javascript_extract_function_name_declaration_form():
    h = JavaScriptHandler()
    assert h.extract_function_name("function greet(name) { return name; }") == "greet"


def test_javascript_extract_function_name_arrow_form():
    h = JavaScriptHandler()
    assert h.extract_function_name("const greet = (name) => name;") == "greet"


def test_javascript_find_call_sites():
    h = JavaScriptHandler()
    source = b"greet('Alice');\nobj.method(1);\n"
    assert h.find_call_sites("greet", source) == ["greet('Alice')"]


def test_typescript_extract_function_name_and_call_sites():
    h = TypeScriptHandler()
    assert h.extract_function_name(
        "function add(a: number, b: number): number { return a + b; }"
    ) == "add"
    source = b"add(2, 3);\n"
    assert h.find_call_sites("add", source) == ["add(2, 3)"]


def test_php_extract_function_name_and_call_sites():
    h = PhpHandler()
    assert h.extract_function_name(
        "function greet(string $name): string { return $name; }"
    ) == "greet"
    source = b"<?php\ngreet('Alice');\n$obj->greet(1);\n"
    assert h.find_call_sites("greet", source) == ["greet('Alice')"]


def test_default_handler_degrades_gracefully_without_queries():
    # Java has no name_query_src/call_query_src configured — must return
    # None/[] rather than raising, so the probe machinery can fall back
    # to LLM-only generation cleanly for languages that don't support this.
    h = JavaHandler()
    assert h.extract_function_name("int add(int a, int b) { return a + b; }") is None
    assert h.find_call_sites("add", b"add(2, 3);") == []
