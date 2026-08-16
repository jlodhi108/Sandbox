import os
import tempfile
from unittest.mock import patch

from benchmark import run_benchmark, print_scorecard


def _touch(path: str, content: str = "def f(x):\n    return x + 1\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_run_benchmark_aggregates_by_language():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "b.js"), "function f(x) { return x + 1; }")

        fake_stats = {
            "a.py": {
                "file_path": os.path.join(root, "a.py"), "language": "python",
                "chunks_total": 3, "chunks_succeeded": 2, "chunks_already_modern": 1,
                "chunks_gave_up": 0, "chunks_punted": 0,
            },
            "b.js": {
                "file_path": os.path.join(root, "b.js"), "language": "javascript",
                "chunks_total": 2, "chunks_succeeded": 1, "chunks_already_modern": 0,
                "chunks_gave_up": 1, "chunks_punted": 0,
            },
        }

        def fake_run_file(file_path, open_pr, max_iterations, **kwargs):
            return fake_stats[os.path.basename(file_path)]

        with patch("benchmark.run_file", side_effect=fake_run_file):
            benchmark = run_benchmark(root, max_iterations=5)

    assert benchmark["by_language"]["python"] == {
        "files": 1, "chunks_total": 3, "chunks_succeeded": 2,
        "chunks_already_modern": 1, "chunks_gave_up": 0, "chunks_punted": 0,
    }
    assert benchmark["by_language"]["javascript"] == {
        "files": 1, "chunks_total": 2, "chunks_succeeded": 1,
        "chunks_already_modern": 0, "chunks_gave_up": 1, "chunks_punted": 0,
    }
    assert benchmark["overall"]["chunks_total"] == 5
    assert benchmark["overall"]["chunks_succeeded"] == 3
    assert benchmark["overall"]["chunks_attempted"] == 4  # 3 succeeded + 1 gave_up, excludes already_modern
    assert benchmark["overall"]["success_rate_of_attempted"] == 0.75


def test_run_benchmark_excludes_already_modern_and_punted_from_attempted():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))

        fake_stats = {
            "a.py": {
                "file_path": os.path.join(root, "a.py"), "language": "python",
                "chunks_total": 5, "chunks_succeeded": 1, "chunks_already_modern": 2,
                "chunks_gave_up": 0, "chunks_punted": 2,
            },
        }

        with patch("benchmark.run_file", side_effect=lambda fp, *a, **k: fake_stats[os.path.basename(fp)]):
            benchmark = run_benchmark(root, max_iterations=5)

    # attempted = succeeded + gave_up = 1 + 0 = 1 — already_modern and
    # punted chunks must NOT inflate the denominator.
    assert benchmark["overall"]["chunks_attempted"] == 1
    assert benchmark["overall"]["success_rate_of_attempted"] == 1.0


def test_run_benchmark_skips_files_that_error_without_crashing():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))

        def fake_run_file(file_path, open_pr, max_iterations, **kwargs):
            raise RuntimeError("boom")

        with patch("benchmark.run_file", side_effect=fake_run_file):
            benchmark = run_benchmark(root, max_iterations=5)

    assert benchmark["by_language"] == {}
    assert benchmark["overall"]["chunks_total"] == 0
    assert benchmark["file_results"][0]["error"] == "boom"


def test_run_benchmark_handles_zero_attempted_chunks():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))

        fake_stats = {
            "a.py": {
                "file_path": os.path.join(root, "a.py"), "language": "python",
                "chunks_total": 1, "chunks_succeeded": 0, "chunks_already_modern": 1,
                "chunks_gave_up": 0, "chunks_punted": 0,
            },
        }
        with patch("benchmark.run_file", side_effect=lambda fp, *a, **k: fake_stats[os.path.basename(fp)]):
            benchmark = run_benchmark(root, max_iterations=5)

    assert benchmark["overall"]["chunks_attempted"] == 0
    assert benchmark["overall"]["success_rate_of_attempted"] is None


def test_print_scorecard_does_not_crash_on_empty_benchmark(capsys):
    empty = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "by_language": {},
        "overall": {"chunks_total": 0, "chunks_succeeded": 0, "chunks_attempted": 0, "success_rate_of_attempted": None},
        "llm_usage": {"total_calls": 0},
        "duration_seconds": 0.0,
    }
    print_scorecard(empty)
    captured = capsys.readouterr()
    assert "Overall: 0/0" in captured.out
    assert "n/a" in captured.out
