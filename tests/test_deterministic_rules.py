import deterministic_rules as dr


def test_js_var_becomes_const_when_never_reassigned():
    result = dr.try_apply("javascript", "function f() {\n  var x = 1;\n  return x + 1;\n}")
    assert "const x = 1;" in result
    assert "var" not in result


def test_js_var_becomes_let_when_reassigned():
    result = dr.try_apply("javascript", "function f() {\n  var x = 1;\n  x = x + 1;\n  return x;\n}")
    assert "let x = 1;" in result
    assert "const" not in result


def test_js_var_becomes_let_when_incremented():
    result = dr.try_apply("javascript", "function f() {\n  var i = 0;\n  i++;\n  return i;\n}")
    assert "let i = 0;" in result


def test_js_var_becomes_let_when_augmented_assigned():
    result = dr.try_apply("javascript", "function f() {\n  var i = 0;\n  i += 1;\n  return i;\n}")
    assert "let i = 0;" in result


def test_js_var_without_initializer_becomes_let_not_const():
    result = dr.try_apply("javascript", "function f() {\n  var x;\n  x = 5;\n  return x;\n}")
    assert "let x;" in result


def test_js_multiple_var_declarations_handled_independently():
    result = dr.try_apply(
        "javascript",
        "function f() {\n  var a = 1;\n  var b = 2;\n  b = b + 1;\n  return a + b;\n}",
    )
    assert "const a = 1;" in result
    assert "let b = 2;" in result


def test_js_comma_declarator_list_aborts_whole_rule():
    code = "function f() {\n  var a = 1, b = 2;\n  return a + b;\n}"
    assert dr.try_apply("javascript", code) is None


def test_js_destructuring_pattern_aborts_whole_rule():
    code = "function f(obj) {\n  var {a, b} = obj;\n  return a + b;\n}"
    assert dr.try_apply("javascript", code) is None


def test_js_no_var_returns_none():
    code = "function f() {\n  let x = 1;\n  return x;\n}"
    assert dr.try_apply("javascript", code) is None


def test_js_applies_to_typescript_too():
    result = dr.try_apply("typescript", "function f(): number {\n  var x = 1;\n  return x;\n}")
    assert "const x = 1;" in result


def test_php_array_becomes_bracket_syntax():
    result = dr.try_apply("php", "function f() {\n    $a = array(1, 2);\n    return $a;\n}")
    assert "[1, 2]" in result
    assert "array(" not in result


def test_php_nested_array_all_converted():
    result = dr.try_apply(
        "php", "function f() {\n    $a = array(1, 2, array(3, 4));\n    return $a;\n}",
    )
    assert result is not None
    assert "array(" not in result
    assert "[1, 2, [3, 4]]" in result


def test_php_already_bracket_syntax_returns_none():
    code = "function f() {\n    $a = [1, 2];\n    return $a;\n}"
    assert dr.try_apply("php", code) is None


def test_try_apply_returns_none_for_unsupported_language():
    assert dr.try_apply("python", "def f(x):\n    return x + 1\n") is None
    assert dr.try_apply("java", "int f(int x) { return x; }") is None
    assert dr.try_apply("cpp", "int f(int x) { return x; }") is None
