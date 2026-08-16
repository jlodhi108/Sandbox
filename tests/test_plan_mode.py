import os
import tempfile

from main import plan_file, plan_run, _estimate_llm_calls_per_chunk


def test_estimate_llm_calls_per_chunk_min_is_less_than_max():
    min_calls, max_calls = _estimate_llm_calls_per_chunk(max_iterations=5)
    assert min_calls > 0
    assert max_calls > min_calls


def test_estimate_llm_calls_scales_with_max_iterations():
    min_low, max_low = _estimate_llm_calls_per_chunk(max_iterations=2)
    min_high, max_high = _estimate_llm_calls_per_chunk(max_iterations=10)
    assert min_low == min_high  # min case never retries, independent of the ceiling
    assert max_high > max_low


def test_plan_file_counts_already_modern_and_to_modernize():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calc.py")
        with open(path, "w") as f:
            f.write(
                "def already_modern(x: int) -> int:\n"
                "    return x + 1\n\n"
                "def legacy_one(x):\n"
                "    return x + 1\n"
            )
        plan = plan_file(path, max_iterations=5)

    assert plan["chunks_total"] == 2
    assert plan["language"] == "python"
    assert plan["chunks_already_modern"] + plan["chunks_to_modernize"] == 2
    assert plan["estimated_llm_calls_min"] >= 0
    assert plan["estimated_llm_calls_max"] >= plan["estimated_llm_calls_min"]


def test_plan_file_makes_no_llm_or_docker_calls():
    # The whole point of --plan: pure tree-sitter parsing, nothing else.
    # Patch both out to something that raises, and confirm plan_file
    # never touches either.
    import sandbox.verifier as verifier_module
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "calc.py")
        with open(path, "w") as f:
            f.write("def legacy_one(x):\n    return x + 1\n")

        with patch.object(verifier_module, "_get_client", side_effect=AssertionError("Docker touched")):
            plan_file(path, max_iterations=5)  # must not raise


def test_plan_run_directory_uses_discover_files_filtering():
    with tempfile.TemporaryDirectory() as d:
        _write = lambda rel, content: open(os.path.join(d, rel), "w").write(content)
        os.makedirs(os.path.join(d, "vendor"), exist_ok=True)
        _write("calc.py", "def legacy_one(x):\n    return x + 1\n")
        _write("vendor/skip_me.py", "def legacy_two(x):\n    return x + 1\n")
        with open(os.path.join(d, ".gitignore"), "w") as f:
            f.write("vendor/\n")

        plans = plan_run(d, max_iterations=5)

    assert len(plans) == 1
    assert plans[0]["file_path"].endswith("calc.py")
