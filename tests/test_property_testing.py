import property_testing as pt


def test_infer_param_strategies_maps_known_types():
    strategies = pt.infer_param_strategies("def add(a: int, b: str) -> str:\n    return b * a")
    assert strategies == {"a": "st.integers()", "b": "st.text()"}


def test_infer_param_strategies_returns_none_without_annotations():
    assert pt.infer_param_strategies("def add(a, b):\n    return a + b") is None


def test_infer_param_strategies_returns_none_for_partial_annotations():
    assert pt.infer_param_strategies("def add(a: int, b):\n    return a + b") is None


def test_infer_param_strategies_returns_none_for_unknown_type():
    code = "def f(x: MyCustomType) -> int:\n    return 1"
    assert pt.infer_param_strategies(code) is None


def test_infer_param_strategies_returns_none_for_varargs():
    assert pt.infer_param_strategies("def f(*args: int) -> int:\n    return sum(args)") is None


def test_infer_param_strategies_returns_none_for_kwargs():
    assert pt.infer_param_strategies("def f(**kwargs: int) -> int:\n    return 1") is None


def test_infer_param_strategies_returns_none_for_keyword_only_args():
    assert pt.infer_param_strategies("def f(*, x: int) -> int:\n    return x") is None


def test_infer_param_strategies_returns_none_for_zero_arg_function():
    assert pt.infer_param_strategies("def f() -> int:\n    return 1") is None


def test_infer_param_strategies_returns_none_for_syntax_error():
    assert pt.infer_param_strategies("def f(x: int\n    return x") is None


def test_generate_property_test_returns_none_when_not_applicable():
    assert pt.generate_property_test("def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b") is None


def test_generate_property_test_produces_runnable_script():
    original = "def add(a: int, b: int) -> int:\n    return a + b"
    modernized = "def add(a: int, b: int) -> int:\n    return a + b"
    script = pt.generate_property_test(original, modernized)
    assert script is not None
    assert "from hypothesis import" in script
    assert "_original_fn" in script and "_modernized_fn" in script
    assert "PROPERTY_TEST_OK" in script
    # The script must embed the exact source via repr(), not raw
    # interpolation — safe against quotes/newlines in the function body
    # and trivially verifiable by round-tripping through exec().
    namespace = {}
    exec(compile(script.split("_test_equivalence()")[0], "<test>", "exec"), namespace)
    assert namespace["_original_fn"](2, 3) == 5
    assert namespace["_modernized_fn"](2, 3) == 5


def test_build_property_test_script_embeds_source_safely_with_special_characters():
    # Function bodies containing quotes/backslashes/newlines must not
    # break out of the exec()'d string — repr() handles this correctly
    # by construction, this proves it end to end.
    original = 'def f(x: str) -> str:\n    return x + "\'quoted\' and \\\\backslash\\\\"'
    script = pt.build_property_test_script("f", original, original, {"x": "st.text()"})
    namespace = {}
    exec(compile(script.split("_test_equivalence()")[0], "<test>", "exec"), namespace)
    assert namespace["_original_fn"]("hi") == "hi'quoted' and \\backslash\\"
