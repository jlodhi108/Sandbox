from regression_tests import (
    generate_regression_test_file, regression_test_filename,
)


def test_regression_test_filename_python():
    assert regression_test_filename("python", "/repo/calc.py") == "/repo/test_calc_modernized.py"


def test_regression_test_filename_javascript():
    assert regression_test_filename("javascript", "/repo/calc.js") == "/repo/calc.modernized.test.js"


def test_regression_test_filename_typescript():
    assert regression_test_filename("typescript", "/repo/calc.ts") == "/repo/calc.modernized.test.ts"


def test_regression_test_filename_php():
    assert regression_test_filename("php", "/repo/calc.php") == "/repo/calcModernizedTest.php"


def test_regression_test_filename_raises_for_unsupported_language():
    import pytest
    with pytest.raises(ValueError):
        regression_test_filename("cpp", "/repo/calc.cpp")


def test_generate_returns_none_when_no_chunk_has_probes():
    chunk_results = [{"modernized_code": "def add(a, b):\n    return a + b", "probes": []}]
    assert generate_regression_test_file("python", chunk_results) is None


def test_generate_returns_none_for_unsupported_language():
    chunk_results = [{
        "modernized_code": "int add(int a, int b) { return a + b; }",
        "probes": [{"snippet": "printf(\"%d\", add(1,2))", "baseline_stdout": "3"}],
    }]
    assert generate_regression_test_file("cpp", chunk_results) is None


def test_generate_python_embeds_function_and_probe_assertions():
    chunk_results = [{
        "modernized_code": "def add(a, b):\n    return a + b",
        "probes": [
            {"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"},
            {"snippet": "print(add(0, 0))", "baseline_stdout": "0\n"},
        ],
    }]
    source = generate_regression_test_file("python", chunk_results)
    assert "def add(a, b):" in source
    assert "import unittest" in source
    assert "class TestModernizedRegressions(unittest.TestCase):" in source
    assert "def test_probe_0(self):" in source
    assert "def test_probe_1(self):" in source
    assert "print(add(2, 3))" in source
    assert repr("5\n") in source

    # Must be valid, executable Python — this is a hard requirement,
    # not just a string-shape check.
    compile(source, "<generated>", "exec")


def test_generate_python_combines_multiple_chunks_from_one_file():
    chunk_results = [
        {"modernized_code": "def add(a, b):\n    return a + b",
         "probes": [{"snippet": "print(add(1, 1))", "baseline_stdout": "2\n"}]},
        {"modernized_code": "def sub(a, b):\n    return a - b",
         "probes": [{"snippet": "print(sub(5, 3))", "baseline_stdout": "2\n"}]},
    ]
    source = generate_regression_test_file("python", chunk_results)
    assert "def add(a, b):" in source
    assert "def sub(a, b):" in source
    assert "def test_probe_0(self):" in source
    assert "def test_probe_1(self):" in source
    compile(source, "<generated>", "exec")


def test_generate_javascript_produces_commonjs_require():
    chunk_results = [{
        "modernized_code": "function add(a, b) {\n  return a + b;\n}",
        "probes": [{"snippet": "console.log(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    source = generate_regression_test_file("javascript", chunk_results)
    assert 'require("node:assert")' in source
    assert "declare function require" not in source  # TS-only, must not leak into JS
    assert "function add(a, b)" in source


def test_generate_typescript_includes_ambient_require_declaration():
    chunk_results = [{
        "modernized_code": "function add(a: number, b: number): number {\n  return a + b;\n}",
        "probes": [{"snippet": "console.log(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    source = generate_regression_test_file("typescript", chunk_results)
    assert "declare function require(name: string): any;" in source


def test_generate_javascript_puts_snippet_with_trailing_comment_on_its_own_line():
    # Real bug caught live: a probe snippet with a trailing `//` comment
    # (the model doesn't always follow "nothing else on that line")
    # embedded on the SAME line as a closing brace comments out that
    # brace too, corrupting the whole file with an unterminated block —
    # confirmed via a real `node` run raising "SyntaxError: Unexpected
    # end of input". The snippet must never share a line with anything
    # meaningful after it.
    chunk_results = [{
        "modernized_code": "function add(a, b) {\n  return a + b;\n}",
        "probes": [{"snippet": 'console.log(add(2, 3)); // "5"', "baseline_stdout": "5\n"}],
    }]
    source = generate_regression_test_file("javascript", chunk_results)
    for line in source.splitlines():
        if 'console.log(add(2, 3)); // "5"' in line:
            assert line.strip() == 'console.log(add(2, 3)); // "5"'
            break
    else:
        raise AssertionError("probe snippet line not found in generated source")


def test_generate_php_puts_snippet_with_trailing_comment_on_its_own_line():
    chunk_results = [{
        "modernized_code": "function greet(string $name): string {\n    return \"Hello, \" . $name;\n}",
        "probes": [{"snippet": "echo greet('Bob'); // Bob", "baseline_stdout": "Hello, Bob"}],
    }]
    source = generate_regression_test_file("php", chunk_results)
    for line in source.splitlines():
        if "echo greet('Bob'); // Bob" in line:
            assert line.strip() == "echo greet('Bob'); // Bob"
            break
    else:
        raise AssertionError("probe snippet line not found in generated source")


def test_generate_php_uses_output_buffering_and_strict_comparison():
    chunk_results = [{
        "modernized_code": "function greet(string $name): string {\n    return \"Hello, \" . $name;\n}",
        "probes": [{"snippet": "echo greet('Bob');", "baseline_stdout": "Hello, Bob"}],
    }]
    source = generate_regression_test_file("php", chunk_results)
    assert source.startswith("<?php")
    assert "ob_start()" in source
    assert "function greet(string $name): string {" in source
    assert "echo greet('Bob');" in source


def test_generate_php_escapes_single_quotes_in_baseline():
    chunk_results = [{
        "modernized_code": "function quote(): string {\n    return \"it's\";\n}",
        "probes": [{"snippet": "echo quote();", "baseline_stdout": "it's"}],
    }]
    source = generate_regression_test_file("php", chunk_results)
    assert "\\'" in source  # the escaped baseline literal
    # Must not contain an unescaped stray quote that would break the PHP
    # string literal it's embedded in. re.DOTALL: the snippet sits on its
    # own line (see the multi-line-embedding comment in
    # regression_tests.py), so this now spans multiple lines.
    import re
    literal_match = re.search(r"_check\(\"probe_0\", _capture\(function\(\) \{.*?\}\), (.+?)\);", source, re.DOTALL)
    assert literal_match is not None
