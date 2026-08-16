"""Property-based equivalence testing for Python chunks, layered on top
of the example-based probes in agents/graph.py (real call sites + LLM-
synthesized examples). Research on testing LLM-generated code found
property-based and example-based testing each independently catch a
majority of bugs, but MORE bugs together than either alone — they miss
different things: example-based probes only ever check the specific
inputs someone (a real call site, or an LLM) happened to think of;
property-based testing samples across the ENTIRE input space a type
signature admits, catching edge cases nobody happened to write down.

Deliberately conservative about when it applies: only chunks with a
FULLY type-hinted parameter list, where every annotation maps to a
known Hypothesis strategy (see _TYPE_TO_STRATEGY), get a property test
generated at all. Anything else — missing annotations, *args/**kwargs,
keyword-only params, an annotation this module doesn't recognize —
returns None rather than guessing at a strategy, the same fail-closed
contract every other "can I safely check this" function in this
codebase uses (see languages/base.py's already_modern, or
agents/nodes.py's check_requires_resolvable).

The generated script is plain, self-contained Python (no dependency on
this project's own modules) that `exec`s both the original and
modernized function source directly — it never touches the LLM, so
there's no hallucination risk here, only the risk that a genuinely
divergent input exists that Hypothesis's random search happens to find
(which is exactly the point)."""
import ast

# Conservative on purpose: adding a type here is a promise that
# EVERY value this strategy can generate is a value the corresponding
# Python type annotation actually permits, and that calling the function
# with it can never raise for a reason unrelated to the modernization
# itself (e.g. int includes negative/zero/huge values on purpose — a
# function that can't handle those either wasn't fully compatible with
# its own type hint before this checked it, or WILL break exactly the
# same way on both original and modernized, which is what "OK"/"EXC"
# symmetric comparison in the generated script is for).
_TYPE_TO_STRATEGY = {
    "int": "st.integers()",
    "float": "st.floats(allow_nan=False, allow_infinity=False)",
    "str": "st.text()",
    "bool": "st.booleans()",
    "bytes": "st.binary()",
    "list[int]": "st.lists(st.integers())",
    "list[str]": "st.lists(st.text())",
    "list[float]": "st.lists(st.floats(allow_nan=False, allow_infinity=False))",
    "dict[str, int]": "st.dictionaries(st.text(), st.integers())",
    "dict[str, str]": "st.dictionaries(st.text(), st.text())",
}


def _parse_function(function_code: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(function_code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def infer_param_strategies(function_code: str) -> dict[str, str] | None:
    """{param_name: hypothesis_strategy_source} for every parameter, or
    None if this chunk can't be safely property-tested (see this
    module's docstring for the exact conditions)."""
    func = _parse_function(function_code)
    if func is None:
        return None
    if func.args.vararg or func.args.kwarg or func.args.kwonlyargs or func.args.posonlyargs:
        return None  # forms this module doesn't attempt to model
    args = func.args.args
    if not args:
        return None  # a zero-arg function has exactly one input — the
        # example-based baseline/probe checks already cover it fully;
        # nothing for random sampling to add.

    strategies = {}
    for arg in args:
        if arg.annotation is None:
            return None
        try:
            type_str = ast.unparse(arg.annotation).strip()
        except Exception:
            return None
        strategy = _TYPE_TO_STRATEGY.get(type_str)
        if strategy is None:
            return None
        strategies[arg.arg] = strategy
    return strategies


def _function_name(function_code: str) -> str | None:
    func = _parse_function(function_code)
    return func.name if func else None


def build_property_test_script(
    function_name: str, original_code: str, modernized_code: str, strategies: dict[str, str],
) -> str:
    """A self-contained Python script (no import of this project's own
    code) that `exec`s both function sources into separate namespaces
    and asserts they agree across many Hypothesis-generated inputs.
    Exceptions are compared BY TYPE, not message — two implementations
    raising the same kind of error for the same bad input is equivalent
    behavior even if the exact wording differs, which matters because a
    modernization legitimately might phrase an error differently (e.g.
    a manual `if x < 0: raise ValueError(...)` becoming a stdlib
    function that raises the same ValueError type with its own message)."""
    fixed_dict_items = ", ".join(f"{name!r}: {strategy}" for name, strategy in strategies.items())
    given_strategy = f"st.fixed_dictionaries({{{fixed_dict_items}}})"
    return (
        "from hypothesis import given, settings, HealthCheck, strategies as st\n\n"
        "_original_ns = {}\n"
        "_modernized_ns = {}\n"
        f"exec({original_code!r}, _original_ns)\n"
        f"exec({modernized_code!r}, _modernized_ns)\n\n"
        f"_original_fn = _original_ns[{function_name!r}]\n"
        f"_modernized_fn = _modernized_ns[{function_name!r}]\n\n\n"
        "def _call(fn, kwargs):\n"
        "    try:\n"
        "        return (\"OK\", fn(**kwargs))\n"
        "    except Exception as e:\n"
        "        return (\"EXC\", type(e).__name__)\n\n\n"
        "@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))\n"
        f"@given(kwargs={given_strategy})\n"
        "def _test_equivalence(kwargs):\n"
        "    original_result = _call(_original_fn, kwargs)\n"
        "    modernized_result = _call(_modernized_fn, kwargs)\n"
        "    assert original_result == modernized_result, (\n"
        "        f\"divergence with args {kwargs!r}: original={original_result!r} \"\n"
        "        f\"modernized={modernized_result!r}\"\n"
        "    )\n\n\n"
        "_test_equivalence()\n"
        "print(\"PROPERTY_TEST_OK\")\n"
    )


def generate_property_test(original_code: str, modernized_code: str) -> str | None:
    """Top-level entry point: build a runnable property-test script for
    this chunk, or None if it can't be safely generated (see
    infer_param_strategies). Strategies are inferred from the ORIGINAL
    signature — the modernized version is trusted to have kept the same
    parameter names/types (a structural mismatch there would already
    have been caught by _validate_single_chunk / the sandbox compile
    step before this ever runs)."""
    strategies = infer_param_strategies(original_code)
    if strategies is None:
        return None
    function_name = _function_name(original_code)
    if function_name is None:
        return None
    return build_property_test_script(function_name, original_code, modernized_code, strategies)
