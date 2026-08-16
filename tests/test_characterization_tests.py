import pytest

from characterization_tests import (
    generate_characterization_test_file, characterization_test_filename,
)


def test_characterization_test_filename_python():
    assert characterization_test_filename("python", "/repo/calc.py") == "/repo/test_calc_characterization.py"


def test_characterization_test_filename_javascript():
    assert characterization_test_filename("javascript", "/repo/calc.js") == "/repo/calc.characterization.test.js"


def test_characterization_test_filename_typescript():
    assert characterization_test_filename("typescript", "/repo/calc.ts") == "/repo/calc.characterization.test.ts"


def test_characterization_test_filename_php():
    assert characterization_test_filename("php", "/repo/calc.php") == "/repo/calcCharacterizationTest.php"


def test_characterization_test_filename_raises_for_unsupported_language():
    with pytest.raises(ValueError):
        characterization_test_filename("cpp", "/repo/calc.cpp")


def test_characterization_filename_never_collides_with_regression_filename():
    from regression_tests import regression_test_filename
    for language, path in [
        ("python", "/repo/calc.py"), ("javascript", "/repo/calc.js"),
        ("typescript", "/repo/calc.ts"), ("php", "/repo/calc.php"),
    ]:
        assert characterization_test_filename(language, path) != regression_test_filename(language, path)


def test_generate_returns_none_when_no_chunk_has_probes():
    chunks = [{"original_code": "def add(a, b):\n    return a + b", "probes": []}]
    assert generate_characterization_test_file("python", chunks) is None


def test_generate_returns_none_for_unsupported_language():
    chunks = [{
        "original_code": "int add(int a, int b) { return a + b; }",
        "probes": [{"snippet": "printf(\"%d\", add(1,2))", "baseline_stdout": "3"}],
    }]
    assert generate_characterization_test_file("cpp", chunks) is None


def test_generate_python_embeds_original_code_not_modernized():
    # The whole point: this pins the ORIGINAL (pre-rewrite) behavior —
    # embedding modernized_code here would defeat the purpose (nothing
    # left to compare a future rewrite attempt against).
    chunks = [{
        "original_code": "def add(a, b):\n    return a+b",  # deliberately legacy-styled
        "modernized_code": "def add(a: int, b: int) -> int:\n    return a + b",  # must NOT appear
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    source = generate_characterization_test_file("python", chunks)
    assert "def add(a, b):\n    return a+b" in source
    assert "def add(a: int, b: int) -> int:" not in source
    compile(source, "<generated>", "exec")


def test_generate_python_uses_characterization_class_name_and_header():
    chunks = [{
        "original_code": "def add(a, b):\n    return a + b",
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    source = generate_characterization_test_file("python", chunks)
    assert "class TestCharacterization(unittest.TestCase):" in source
    assert "--characterize" in source
    assert "ORIGINAL" in source


def test_generate_javascript_embeds_original_code():
    chunks = [{
        "original_code": "function add(a, b) {\n  return a + b;\n}",
        "modernized_code": "const add = (a, b) => a + b;",
        "probes": [{"snippet": "console.log(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    source = generate_characterization_test_file("javascript", chunks)
    assert "function add(a, b)" in source
    assert "const add = (a, b) => a + b;" not in source


def test_generate_php_embeds_original_code():
    chunks = [{
        "original_code": "function add($a, $b) {\n    return $a + $b;\n}",
        "modernized_code": "function add(int $a, int $b): int {\n    return $a + $b;\n}",
        "probes": [{"snippet": "echo add(2, 3);", "baseline_stdout": "5"}],
    }]
    source = generate_characterization_test_file("php", chunks)
    assert "function add($a, $b)" in source
    assert "function add(int $a, int $b): int" not in source


def test_generate_combines_multiple_chunks_including_ones_that_gave_up():
    # Characterization tests must be generatable for chunks the
    # modernizer FAILED on too — they still have probes captured
    # against the original before any attempt was made.
    chunks = [
        {"original_code": "def add(a, b):\n    return a + b",
         "probes": [{"snippet": "print(add(1, 1))", "baseline_stdout": "2\n"}]},
        {"original_code": "def risky(x):\n    return weird_legacy_call(x)",  # this one "gave up"
         "probes": [{"snippet": "print(risky(1))", "baseline_stdout": "1\n"}]},
    ]
    source = generate_characterization_test_file("python", chunks)
    assert "def add(a, b):" in source
    assert "def risky(x):" in source
    compile(source, "<generated>", "exec")
