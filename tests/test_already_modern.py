from languages.cpp import CppHandler
from languages.python_lang import PythonHandler
from languages.javascript_lang import JavaScriptHandler, TypeScriptHandler
from languages.java_lang import JavaHandler
from languages.php_lang import PhpHandler


def test_cpp_detects_raw_new_and_delete_as_legacy():
    h = CppHandler()
    assert h.already_modern("int* create_array(int size) { return new int[size]; }") is False
    assert h.already_modern("void free_array(int* arr) { delete[] arr; }") is False
    assert h.already_modern("void free_array(int* arr) { delete arr; }") is False


def test_cpp_word_boundary_avoids_false_positive_on_renew():
    h = CppHandler()
    # "renew" contains "new" as a substring but is not the `new` keyword
    assert h.already_modern("void renew() { std::cout << 1; }") is True


def test_cpp_clean_function_is_already_modern():
    h = CppHandler()
    assert h.already_modern("void print() { std::cout << id << std::endl; }") is True


def test_python_detects_percent_formatting_as_legacy():
    h = PythonHandler()
    assert h.already_modern('def greet(name):\n    return "Hello, %s!" % name') is False


def test_python_detects_dot_format_as_legacy():
    h = PythonHandler()
    assert h.already_modern('def greet(name):\n    return "Hello, {}!".format(name)') is False


def test_python_clean_function_is_already_modern():
    h = PythonHandler()
    assert h.already_modern('def greet(name: str) -> str:\n    return f"Hello, {name}!"') is True


def test_javascript_old_function_declaration_is_not_already_modern():
    # Confirmed bug during development: a "no var" only check flagged this
    # as already modern even though it clearly benefits from arrow-function
    # conversion. Must NOT be skipped.
    h = JavaScriptHandler()
    assert h.already_modern('function greet(name) { return "Hello, " + name + "!"; }') is False


def test_javascript_arrow_form_with_no_var_is_already_modern():
    h = JavaScriptHandler()
    assert h.already_modern("const greet = (name) => `Hello, ${name}!`;") is True


def test_javascript_arrow_form_with_var_inside_is_not_modern():
    h = JavaScriptHandler()
    assert h.already_modern("const f = () => { var x = 1; return x; };") is False


def test_typescript_same_rules_as_javascript():
    h = TypeScriptHandler()
    assert h.already_modern("function add(a: number, b: number): number { return a + b; }") is False
    assert h.already_modern("const add = (a: number, b: number): number => a + b;") is True


def test_java_never_skips_by_default():
    h = JavaHandler()
    assert h.already_modern("public static int add(int a, int b) { return a + b; }") is False


def test_php_detects_missing_return_type_as_legacy():
    h = PhpHandler()
    assert h.already_modern('function greet($name) { return "Hello, " . $name; }') is False


def test_php_return_type_present_is_already_modern():
    h = PhpHandler()
    assert h.already_modern('function greet(string $name): string { return "Hello, " . $name; }') is True
