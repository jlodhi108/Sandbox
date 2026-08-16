import json
import os
import tempfile
import time
from unittest.mock import patch

from main import (
    discover_files, write_report, _open_combined_pr, _unified_diff, _run_files_concurrently,
    _isolate_probe_baselines,
)
from languages.python_lang import PythonHandler


def _touch(path: str, content: str = "x = 1\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_isolate_probe_baselines_replaces_whole_file_contaminated_baseline():
    # Real bug caught live: _verify_candidate's recorded baseline_stdout
    # for a probe is "whole file (including its own top-level prints) +
    # probe appended", NOT the probe's own isolated output — confirmed by
    # a real run producing baseline_stdout='5\n5\n' for a probe that only
    # prints once. A durable regression test must isolate the function,
    # so this re-derives a clean baseline via one extra verify() call per
    # probe rather than trusting the recorded (contaminated) one.
    chunk_results = [{
        "modernized_code": "def add(a, b):\n    return a + b",
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n5\n"}],  # contaminated
    }]
    with patch("main.verify") as mock_verify:
        mock_verify.return_value = {"status": "success", "stdout": "5\n", "stderr": "", "exit_code": 0}
        isolated = _isolate_probe_baselines(PythonHandler(), chunk_results)

    assert isolated[0]["probes"][0]["baseline_stdout"] == "5\n"  # the CLEAN isolated value, not "5\n5\n"
    # Confirms it ran ONLY the function + probe, not the whole original file.
    candidate_arg = mock_verify.call_args[0][0]
    assert candidate_arg == "def add(a, b):\n    return a + b\nprint(add(2, 3))\n"


def test_isolate_probe_baselines_drops_probes_that_fail_in_isolation():
    # A probe that fails when re-run in isolation produces no usable
    # baseline for a regression test — dropped rather than embedding a
    # broken/empty expectation.
    chunk_results = [{
        "modernized_code": "def add(a, b):\n    return a + b",
        "probes": [{"snippet": "print(add(2, 3))", "baseline_stdout": "5\n"}],
    }]
    with patch("main.verify") as mock_verify:
        mock_verify.return_value = {"status": "failed", "stdout": "", "stderr": "boom", "exit_code": 1}
        isolated = _isolate_probe_baselines(PythonHandler(), chunk_results)

    assert isolated[0]["probes"] == []


def test_isolate_probe_baselines_is_a_noop_for_chunks_with_no_probes():
    chunk_results = [{"modernized_code": "int add(int a, int b) { return a + b; }", "probes": []}]
    with patch("main.verify") as mock_verify:
        isolated = _isolate_probe_baselines(PythonHandler(), chunk_results)

    assert isolated[0]["probes"] == []
    mock_verify.assert_not_called()


def test_resolve_interactive_review_approves_and_promotes_to_success():
    import main

    final_state = {
        "status": "awaiting_review",
        "review_thread_id": "fake-thread-id",
        "modernized_code": "def add(a, b):\n    return a + b",
        "risk_flag": True,
        "risk_reason": "touches global state",
        "security_flag": False,
        "security_findings": [],
        "mutation_confidence_flag": False,
        "mutation_confidence_reason": "",
        "compiler_stderr": "",
    }
    with patch("main.resume_review", return_value={"status": "approved"}) as mock_resume, \
         patch("builtins.input", return_value="y"):
        resolved = main._resolve_interactive_review(final_state)

    assert resolved["status"] == "success"
    mock_resume.assert_called_once_with("fake-thread-id", approved=True)


def test_resolve_interactive_review_rejects_and_demotes_to_gave_up():
    import main

    final_state = {
        "status": "awaiting_review",
        "review_thread_id": "fake-thread-id",
        "modernized_code": "def add(a, b):\n    return a + b",
        "risk_flag": True,
        "risk_reason": "touches global state",
        "security_flag": False,
        "security_findings": [],
        "mutation_confidence_flag": False,
        "mutation_confidence_reason": "",
        "compiler_stderr": "",
    }
    with patch("main.resume_review", return_value={"status": "rejected"}) as mock_resume, \
         patch("builtins.input", return_value="n"):
        resolved = main._resolve_interactive_review(final_state)

    assert resolved["status"] == "gave_up"
    assert "Rejected during interactive review" in resolved["compiler_stderr"]
    mock_resume.assert_called_once_with("fake-thread-id", approved=False)


def test_resolve_interactive_review_treats_anything_but_y_as_rejection():
    import main

    final_state = {
        "status": "awaiting_review", "review_thread_id": "fake-thread-id",
        "modernized_code": "x", "risk_flag": True, "risk_reason": "x",
        "security_flag": False, "security_findings": [],
        "mutation_confidence_flag": False, "mutation_confidence_reason": "",
        "compiler_stderr": "",
    }
    with patch("main.resume_review", return_value={"status": "rejected"}), \
         patch("builtins.input", return_value=""):  # bare Enter, no answer
        resolved = main._resolve_interactive_review(final_state)

    assert resolved["status"] == "gave_up"


def test_run_repo_refuses_interactive_with_concurrent_workers():
    import main
    import pytest as _pytest
    with _pytest.raises(ValueError, match="workers"):
        main.run_repo("/fake/root", open_pr=False, max_iterations=5, workers=2, interactive=True)


def test_run_repo_refuses_isolate_workers_without_concurrent_workers():
    import main
    import pytest as _pytest
    with _pytest.raises(ValueError, match="--workers > 1"):
        main.run_repo("/fake/root", open_pr=False, max_iterations=5, workers=1, isolate_workers=True)


def test_run_repo_refuses_isolate_workers_on_non_git_directory():
    import main
    import pytest as _pytest
    with tempfile.TemporaryDirectory() as root:
        with _pytest.raises(ValueError, match="git repository"):
            main.run_repo(root, open_pr=False, max_iterations=5, workers=2, isolate_workers=True)


def test_run_repo_refuses_staged_only_on_non_git_directory():
    import main
    import pytest as _pytest
    with tempfile.TemporaryDirectory() as root:
        with _pytest.raises(ValueError, match="git repository"):
            main.run_repo(root, open_pr=False, max_iterations=5, staged_only=True)


def test_run_repo_staged_only_discovers_only_staged_files():
    import subprocess
    import main

    with tempfile.TemporaryDirectory() as root:
        subprocess.run(["git", "init", "-q", root], check=True)
        subprocess.run(["git", "-C", root, "config", "user.email", "t@example.com"], check=True)
        subprocess.run(["git", "-C", root, "config", "user.name", "T"], check=True)
        _touch(os.path.join(root, "staged.py"), "def f(x):\n    return '%s' % x\n")
        _touch(os.path.join(root, "unstaged.py"), "def g(x):\n    return '%s' % x\n")
        subprocess.run(["git", "-C", root, "add", "staged.py"], check=True)

        with patch("main.run_file") as mock_run_file:
            mock_run_file.return_value = {"file_path": "staged.py", "chunks_succeeded": 0}
            main.run_repo(root, open_pr=False, max_iterations=5, staged_only=True)

        processed_paths = [call.args[0] for call in mock_run_file.call_args_list]
        assert processed_paths == [os.path.join(root, "staged.py")]


def test_run_file_in_worktree_copies_output_back_to_the_real_tree():
    import subprocess
    from main import _run_file_in_worktree

    with tempfile.TemporaryDirectory() as root:
        subprocess.run(["git", "init", "-q", root], check=True)
        subprocess.run(["git", "-C", root, "config", "user.email", "t@example.com"], check=True)
        subprocess.run(["git", "-C", root, "config", "user.name", "T"], check=True)
        _touch(os.path.join(root, "calc.py"), "def add(a, b):\n    return a + b\n")
        subprocess.run(["git", "-C", root, "add", "."], check=True)
        subprocess.run(["git", "-C", root, "commit", "-q", "-m", "initial"], check=True)

        def fake_run_file(file_path, open_pr, max_iterations, standalone_pr=True, sibling_sources=None,
                           generate_regression_tests=False, interactive=False, recipe_instruction=None,
                           **kwargs):
            # Simulate run_file() writing its output INSIDE the worktree
            # it was handed (file_path points there, not at root_dir).
            output_path = file_path.replace("calc.py", "calc.modernized.py")
            with open(output_path, "w") as f:
                f.write("def add(a: int, b: int) -> int:\n    return a + b\n")
            return {"file_path": file_path, "output_path": output_path, "chunks_succeeded": 1}

        with patch("main.run_file", side_effect=fake_run_file):
            stats = _run_file_in_worktree(
                root, os.path.join(root, "calc.py"), open_pr=False, max_iterations=5,
                sibling_sources=[], generate_regression_tests=False, recipe_instruction=None,
            )

        # file_path/output_path must be reported relative to the REAL
        # tree, not the throwaway worktree, and the output must actually
        # exist there (copied back, not left stranded in the worktree).
        assert stats["file_path"] == os.path.join(root, "calc.py")
        assert stats["output_path"] == os.path.join(root, "calc.modernized.py")
        assert os.path.isfile(stats["output_path"])
        with open(stats["output_path"]) as f:
            assert "int" in f.read()

        # The worktree itself must be cleaned up afterward.
        listing = subprocess.run(
            ["git", "-C", root, "worktree", "list"], capture_output=True, text=True,
        ).stdout
        assert listing.strip().count("\n") == 0  # only the main worktree line, no extra worktrees


def test_discover_files_finds_supported_extensions():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "b.cpp"))
        _touch(os.path.join(root, "readme.md"))  # unsupported, should be skipped
        found = discover_files(root)
        names = sorted(os.path.basename(f) for f in found)
        assert names == ["a.py", "b.cpp"]


def test_discover_files_skips_excluded_dirs():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "src", "a.py"))
        _touch(os.path.join(root, "node_modules", "vendor.js"))
        _touch(os.path.join(root, ".git", "hooks.py"))
        _touch(os.path.join(root, "venv", "lib.py"))
        found = discover_files(root)
        assert len(found) == 1
        assert found[0].endswith("src/a.py") or found[0].endswith("src\\a.py")


def test_discover_files_skips_own_modernized_output():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "legacy.py"))
        _touch(os.path.join(root, "legacy.modernized.py"))
        found = discover_files(root)
        names = [os.path.basename(f) for f in found]
        assert names == ["legacy.py"]


def test_discover_files_recurses_into_subdirectories():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "nested", "deep", "b.js"))
        found = discover_files(root)
        assert len(found) == 2


def test_discover_files_honors_gitignore_over_hardcoded_list():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "vendor", "b.py"))  # excluded by .gitignore
        _touch(os.path.join(root, "thing.generated.py"))  # excluded by .gitignore pattern
        # "scratch/" is NOT in the hardcoded fallback list, but a real
        # project might still want it excluded via its own .gitignore —
        # proves .gitignore is actually being read, not just the
        # hardcoded fallback list underneath it.
        _touch(os.path.join(root, "scratch", "c.py"))
        _touch(os.path.join(root, ".gitignore"), "vendor/\n*.generated.py\nscratch/\n")

        found = discover_files(root)
        names = sorted(os.path.basename(f) for f in found)
        assert names == ["a.py"]


def test_discover_files_gitignore_negation_still_included():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, ".gitignore"), "*.py\n!keep_me.py\n")
        _touch(os.path.join(root, "ignored.py"))
        _touch(os.path.join(root, "keep_me.py"))
        found = discover_files(root)
        names = [os.path.basename(f) for f in found]
        assert names == ["keep_me.py"]


def test_discover_files_always_excludes_git_dir_even_without_gitignore():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, ".git", "hooks", "pre-commit.py"))
        found = discover_files(root)
        names = [os.path.basename(f) for f in found]
        assert names == ["a.py"]


def test_discover_files_falls_back_to_hardcoded_list_without_gitignore():
    # No .gitignore present at all — must still exclude the standard
    # vendor/build dirs via the hardcoded fallback list.
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "node_modules", "vendor.js"))
        found = discover_files(root)
        names = [os.path.basename(f) for f in found]
        assert names == ["a.py"]


def test_write_report_produces_valid_json_with_expected_shape():
    with tempfile.TemporaryDirectory() as root:
        report_path = os.path.join(root, "report.json")
        fake_results = [
            {"file_path": "a.py", "language": "python", "chunks_succeeded": 2, "chunks_total": 3},
        ]
        write_report(report_path, "file", fake_results)

        with open(report_path) as f:
            data = json.load(f)

        assert data["mode"] == "file"
        assert "generated_at" in data
        assert data["results"] == fake_results


def test_open_combined_pr_includes_only_passing_files():
    with tempfile.TemporaryDirectory() as root:
        passing_output = os.path.join(root, "a.modernized.py")
        _touch(passing_output, "def add(a, b):\n    return a + b\n")
        failing_output = os.path.join(root, "b.modernized.py")
        _touch(failing_output, "def broken():\n    ...\n")

        all_stats = [
            {
                "file_path": "a.py", "language": "python", "output_path": passing_output,
                "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
            },
            {
                "file_path": "b.py", "language": "python", "output_path": failing_output,
                "final_check_passed": False, "chunks_succeeded": 1, "risk_flagged": [],
            },
        ]

        # Track-record eligibility has its own dedicated tests
        # (test_track_record.py) — bypass it here so this test stays
        # focused on "only passing files get included."
        with patch("main.open_multi_file_pr") as mock_pr, \
             patch("main.is_eligible", return_value=(True, "eligible")):
            mock_pr.return_value = "https://github.com/fake/repo/pull/1"
            url = _open_combined_pr(root, all_stats)

        assert url == "https://github.com/fake/repo/pull/1"
        call_kwargs = mock_pr.call_args[1]
        files_included = [f[0] for f in call_kwargs["files"]]
        assert files_included == ["a.py"]  # b.py excluded — failed its check
        assert "1 file(s) included" in call_kwargs["pr_body"]
        assert "1 file(s) skipped" in call_kwargs["pr_body"]


def test_open_combined_pr_returns_none_when_nothing_passed():
    all_stats = [
        {"file_path": "a.py", "output_path": None, "final_check_passed": None},
    ]
    with patch("main.open_multi_file_pr") as mock_pr:
        url = _open_combined_pr("/fake/root", all_stats)
    assert url is None
    mock_pr.assert_not_called()


def test_open_combined_pr_refuses_on_target_test_regression():
    # --run-target-tests found the target repo's OWN pre-existing test
    # suite passed before modernization and doesn't after — an
    # independent, human-authored oracle catching a real regression
    # must refuse the PR outright, same as final_check_passed=False,
    # even if every individual file's OWN checks looked clean.
    import main
    with tempfile.TemporaryDirectory() as root:
        output = os.path.join(root, "a.modernized.py")
        _touch(output, "def add(a, b):\n    return a + b\n")
        all_stats = [{
            "file_path": "a.py", "language": "python", "output_path": output,
            "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
        }]

        backup = main._last_target_test_result
        try:
            main._last_target_test_result = {
                "framework": "pytest", "baseline": {"status": "success"},
                "after": {"status": "failed"}, "regressed": True,
            }
            with patch("main.open_multi_file_pr") as mock_pr, \
                 patch("main.is_eligible", return_value=(True, "eligible")):
                url = _open_combined_pr(root, all_stats)
        finally:
            main._last_target_test_result = backup

    assert url is None
    mock_pr.assert_not_called()


def test_open_combined_pr_proceeds_when_no_regression_detected():
    import main
    with tempfile.TemporaryDirectory() as root:
        output = os.path.join(root, "a.modernized.py")
        _touch(output, "def add(a, b):\n    return a + b\n")
        all_stats = [{
            "file_path": "a.py", "language": "python", "output_path": output,
            "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
        }]

        backup = main._last_target_test_result
        try:
            main._last_target_test_result = {
                "framework": "pytest", "baseline": {"status": "success"},
                "after": {"status": "success"}, "regressed": False,
            }
            with patch("main.open_multi_file_pr") as mock_pr, \
                 patch("main.is_eligible", return_value=(True, "eligible")):
                mock_pr.return_value = "https://github.com/fake/repo/pull/2"
                url = _open_combined_pr(root, all_stats)
        finally:
            main._last_target_test_result = backup

    assert url == "https://github.com/fake/repo/pull/2"


def test_unified_diff_shows_the_actual_change():
    original = "def add(a, b):\n    return a + b\n"
    modernized = "def add(a: int, b: int) -> int:\n    return a + b\n"
    diff = _unified_diff(original, modernized, "util.py")
    assert "-def add(a, b):" in diff
    assert "+def add(a: int, b: int) -> int:" in diff
    assert "util.py (before)" in diff
    assert "util.py (after)" in diff


def test_unified_diff_empty_when_no_change():
    same = "def add(a, b):\n    return a + b\n"
    diff = _unified_diff(same, same, "util.py")
    assert diff == ""


def test_unified_diff_separates_last_line_without_trailing_newline():
    # Regression: chunk text from byte-slicing never has a trailing
    # newline. Without normalizing that before diffing, the last -/+
    # line pair prints with no line break between them at all, e.g.
    # "-}+};" glued together — confirmed live against a real chunk.
    original = "function f() {\n    return 1;\n}"       # no trailing \n
    modernized = "const f = () => {\n    return 1;\n};"  # no trailing \n
    diff = _unified_diff(original, modernized, "f.js")
    lines = diff.splitlines()
    assert "-}" in lines
    assert "+};" in lines
    # must be on separate lines, not glued into one
    assert not any("-}+" in line or "}+};" in line for line in lines)


def test_run_files_concurrently_preserves_original_order_despite_out_of_order_completion():
    # file "a" is slow, file "b" and "c" are fast — they'll finish before
    # "a" does, but results must still come back in [a, b, c] order so
    # --report output is deterministic regardless of scheduling.
    files = ["a.py", "b.py", "c.py"]
    delays = {"a.py": 0.05, "b.py": 0.0, "c.py": 0.0}

    def fake_run_file(file_path, open_pr, max_iterations, standalone_pr, sibling_sources=None, generate_regression_tests=False, interactive=False, recipe_instruction=None, *args, **kwargs):
        time.sleep(delays[file_path])
        return {"file_path": file_path, "chunks_succeeded": 1}

    with patch("main.run_file", side_effect=fake_run_file):
        results = _run_files_concurrently(files, open_pr=False, max_iterations=5, workers=3, file_contents={})

    assert [r["file_path"] for r in results] == ["a.py", "b.py", "c.py"]


def test_run_files_concurrently_isolates_one_file_erroring():
    files = ["good.py", "bad.py"]

    def fake_run_file(file_path, open_pr, max_iterations, standalone_pr, sibling_sources=None, generate_regression_tests=False, interactive=False, recipe_instruction=None, *args, **kwargs):
        if file_path == "bad.py":
            raise RuntimeError("boom")
        return {"file_path": file_path, "chunks_succeeded": 1}

    with patch("main.run_file", side_effect=fake_run_file):
        results = _run_files_concurrently(files, open_pr=False, max_iterations=5, workers=2, file_contents={})

    assert results[0] == {"file_path": "good.py", "chunks_succeeded": 1}
    assert results[1]["file_path"] == "bad.py"
    assert "boom" in results[1]["error"]


def test_open_combined_pr_excludes_language_without_track_record():
    # This is the actual gate, not the bypassed version above — a
    # language with no history at all must be excluded even though its
    # OWN final check passed cleanly.
    with tempfile.TemporaryDirectory() as root:
        py_output = os.path.join(root, "a.modernized.py")
        _touch(py_output, "def add(a, b):\n    return a + b\n")

        all_stats = [
            {
                "file_path": "a.py", "language": "python", "output_path": py_output,
                "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
            },
        ]

        with patch("main.open_multi_file_pr") as mock_pr, \
             patch("main._history", {}):  # no track record for python at all
            url = _open_combined_pr(root, all_stats)

        assert url is None
        mock_pr.assert_not_called()


def test_open_combined_pr_includes_only_the_eligible_language():
    with tempfile.TemporaryDirectory() as root:
        py_output = os.path.join(root, "a.modernized.py")
        _touch(py_output, "def add(a, b):\n    return a + b\n")
        php_output = os.path.join(root, "b.modernized.php")
        _touch(php_output, "<?php\nfunction add() {}\n")

        all_stats = [
            {
                "file_path": "a.py", "language": "python", "output_path": py_output,
                "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
            },
            {
                "file_path": "b.php", "language": "php", "output_path": php_output,
                "final_check_passed": True, "chunks_succeeded": 1, "risk_flagged": [],
            },
        ]
        # python has a strong proven track record; php has none yet
        fake_history = {"python": {"chunks_succeeded": 20, "chunks_attempted": 20}}

        with patch("main.open_multi_file_pr") as mock_pr, \
             patch("main._history", fake_history):
            mock_pr.return_value = "https://github.com/fake/repo/pull/2"
            url = _open_combined_pr(root, all_stats)

        assert url == "https://github.com/fake/repo/pull/2"
        files_included = [f[0] for f in mock_pr.call_args[1]["files"]]
        assert files_included == ["a.py"]  # only python, php excluded


def test_run_file_counts_punted_chunks_separately_from_gave_up():
    # Punted chunks must NOT inflate chunks_gave_up (that counter feeds
    # track_record.py's success-rate calc — a chunk never attempted
    # shouldn't count as a failed attempt any more than an already-modern
    # chunk does).
    import main

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        _touch(file_path, "def legacy_one(x):\n    return '%s' % x\n")

        punted_state = {
            "status": "gave_up", "punted": True, "iteration_count": 0,
            "compiler_stderr": "Punted before attempting (pre-rewrite confidence check): unsure",
            "modernized_code": "", "required_imports": [], "probes": [],
            "used_escalation": False, "used_deterministic_rule": False,
            "risk_flag": False, "security_flag": False, "mutation_confidence_flag": False,
        }
        with patch("main.modernize", return_value=punted_state):
            stats = main.run_file(file_path, open_pr=False, max_iterations=5, punt_check_enabled=True)

    assert stats["chunks_punted"] == 1
    assert stats["chunks_gave_up"] == 0
    assert stats["chunk_details"][0]["punted"] is True


def test_run_file_characterize_writes_test_pinning_original_code():
    import main

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        _touch(file_path, "def greet(name):\n    return '%s' % name\n")

        fake_state = {
            "status": "success", "iteration_count": 1,
            "modernized_code": "def greet(name: str) -> str:\n    return f'{name}'\n",
            "required_imports": [], "probes": [{"snippet": "print(greet('Bob'))", "baseline_stdout": "Bob\n"}],
            "used_escalation": False, "used_deterministic_rule": False, "punted": False,
            "risk_flag": False, "security_flag": False, "mutation_confidence_flag": False,
            "compiler_stderr": "",
        }
        with patch("main.modernize", return_value=fake_state), \
             patch("main.verify", return_value={"status": "success", "stdout": "Bob\n", "stderr": "", "exit_code": 0}):
            stats = main.run_file(file_path, open_pr=False, max_iterations=5, characterize=True)

        assert stats["characterization_test_path"] is not None
        with open(stats["characterization_test_path"]) as f:
            content = f.read()

    assert "def greet(name):" in content  # ORIGINAL code
    assert "def greet(name: str) -> str:" not in content  # NOT modernized code
    compile(content, "<generated>", "exec")


def test_run_file_characterize_captures_gave_up_chunks_too():
    # The safety net matters MOST for chunks this project couldn't
    # handle — those must still get characterized, not skipped.
    import main

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        _touch(file_path, "def risky(x):\n    return '%s' % weird_legacy_call(x)\n")

        fake_state = {
            "status": "gave_up", "punted": False, "iteration_count": 5,
            "modernized_code": "", "required_imports": [],
            "probes": [{"snippet": "print(risky(1))", "baseline_stdout": "1\n"}],
            "used_escalation": False, "used_deterministic_rule": False,
            "risk_flag": False, "security_flag": False, "mutation_confidence_flag": False,
            "compiler_stderr": "gave up after 5 attempts",
        }
        with patch("main.modernize", return_value=fake_state), \
             patch("main.verify", return_value={"status": "success", "stdout": "1\n", "stderr": "", "exit_code": 0}):
            stats = main.run_file(file_path, open_pr=False, max_iterations=5, characterize=True)

        assert stats["characterization_test_path"] is not None
        with open(stats["characterization_test_path"]) as f:
            content = f.read()

    assert "def risky(x):" in content


def test_run_file_characterize_off_by_default():
    import main

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        _touch(file_path, "def greet(name):\n    return '%s' % name\n")

        fake_state = {
            "status": "success", "iteration_count": 1,
            "modernized_code": "def greet(name: str) -> str:\n    return f'{name}'\n",
            "required_imports": [], "probes": [{"snippet": "print(greet('Bob'))", "baseline_stdout": "Bob\n"}],
            "used_escalation": False, "used_deterministic_rule": False, "punted": False,
            "risk_flag": False, "security_flag": False, "mutation_confidence_flag": False,
            "compiler_stderr": "",
        }
        with patch("main.modernize", return_value=fake_state), \
             patch("main.verify", return_value={"status": "success", "stdout": "Bob\n", "stderr": "", "exit_code": 0}):
            stats = main.run_file(file_path, open_pr=False, max_iterations=5)

    assert stats["characterization_test_path"] is None


def test_run_file_computes_correct_start_line_for_security_flagged_chunk():
    # security_flagged's start_line must be the REAL file's line number
    # where the chunk begins, not byte 0 — sarif_report.py depends on
    # this to report accurate line numbers.
    import main

    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "calc.py")
        # 3 blank-ish leading lines before the flagged function, so its
        # real start line is 4, not 1.
        _touch(file_path, "# comment 1\n# comment 2\n# comment 3\n\ndef greet(name):\n    return '%s' % name\n")

        fake_state = {
            "status": "success", "iteration_count": 1,
            "modernized_code": "def greet(name):\n    import os\n    os.system(name)\n",
            "required_imports": [], "probes": [],
            "used_escalation": False, "used_deterministic_rule": False, "punted": False,
            "risk_flag": False, "security_flag": True,
            "security_findings": [{"rule_id": "py-os-system", "line": 2, "message": "danger"}],
            "mutation_confidence_flag": False,
            "compiler_stderr": "",
        }
        with patch("main.modernize", return_value=fake_state), \
             patch("main.verify", return_value={"status": "success", "stdout": "", "stderr": "", "exit_code": 0}):
            stats = main.run_file(file_path, open_pr=False, max_iterations=5)

    assert len(stats["security_flagged"]) == 1
    assert stats["security_flagged"][0]["start_line"] == 5  # 1-indexed line "def greet" starts on
