import json
import os
import tempfile
import time
from unittest.mock import patch

from main import discover_files, write_report, _open_combined_pr, _unified_diff, _run_files_concurrently


def _touch(path: str, content: str = "x = 1\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


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

    def fake_run_file(file_path, open_pr, max_iterations, standalone_pr, sibling_sources=None):
        time.sleep(delays[file_path])
        return {"file_path": file_path, "chunks_succeeded": 1}

    with patch("main.run_file", side_effect=fake_run_file):
        results = _run_files_concurrently(files, open_pr=False, max_iterations=5, workers=3, file_contents={})

    assert [r["file_path"] for r in results] == ["a.py", "b.py", "c.py"]


def test_run_files_concurrently_isolates_one_file_erroring():
    files = ["good.py", "bad.py"]

    def fake_run_file(file_path, open_pr, max_iterations, standalone_pr, sibling_sources=None):
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
