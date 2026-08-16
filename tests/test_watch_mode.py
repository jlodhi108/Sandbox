import os
import tempfile
from unittest.mock import patch

from main import _scan_mtimes, _diff_changed_files, watch_run


def _touch(path: str, content: str = "def f(x):\n    return x + 1\n") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_scan_mtimes_only_includes_modernizable_files():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "readme.md"), "not code")
        mtimes = _scan_mtimes(root)
        names = {os.path.basename(f) for f in mtimes}
        assert names == {"a.py"}


def test_scan_mtimes_excludes_own_modernized_output():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        _touch(os.path.join(root, "a.modernized.py"))
        mtimes = _scan_mtimes(root)
        names = {os.path.basename(f) for f in mtimes}
        assert names == {"a.py"}


def test_diff_changed_files_detects_new_file():
    old = {"a.py": 100.0}
    new = {"a.py": 100.0, "b.py": 200.0}
    assert _diff_changed_files(old, new) == ["b.py"]


def test_diff_changed_files_detects_modified_mtime():
    old = {"a.py": 100.0}
    new = {"a.py": 150.0}
    assert _diff_changed_files(old, new) == ["a.py"]


def test_diff_changed_files_ignores_unchanged_files():
    old = {"a.py": 100.0, "b.py": 200.0}
    new = {"a.py": 100.0, "b.py": 200.0}
    assert _diff_changed_files(old, new) == []


def test_diff_changed_files_drops_deleted_files_silently():
    old = {"a.py": 100.0, "b.py": 200.0}
    new = {"a.py": 100.0}
    assert _diff_changed_files(old, new) == []


def test_watch_run_processes_a_file_edited_mid_watch():
    with tempfile.TemporaryDirectory() as root:
        file_path = os.path.join(root, "a.py")
        _touch(file_path)

        processed = []

        def fake_run_file(fp, open_pr, max_iterations, standalone_pr=True, sibling_sources=None,
                           generate_regression_tests=False, interactive=False, recipe_instruction=None,
                           **kwargs):
            processed.append(fp)
            return {"file_path": fp, "chunks_succeeded": 1}

        def fake_sleep(_seconds):
            # Edit the file mid-"sleep" so the NEXT scan sees a changed
            # mtime. patch() replaces the SAME shared `time` module object
            # main.py imports, so calling the real time.sleep here isn't
            # possible without capturing it first — set the mtime forward
            # explicitly instead of relying on real wall-clock time to pass.
            if len(processed) == 0:
                _touch(file_path, "def f(x):\n    return x + 2\n")
                future = os.path.getmtime(file_path) + 10
                os.utime(file_path, (future, future))

        with patch("main.run_repo") as mock_run_repo, \
             patch("main.run_file", side_effect=fake_run_file), \
             patch("main.time.sleep", side_effect=fake_sleep):
            watch_run(root, open_pr=False, max_iterations=5, initial_pass=True, _max_polls=2)

        mock_run_repo.assert_called_once()  # the initial full pass
        assert processed == [file_path]


def test_watch_run_skips_initial_pass_when_disabled():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        with patch("main.run_repo") as mock_run_repo, \
             patch("main.time.sleep"):
            watch_run(root, open_pr=False, max_iterations=5, initial_pass=False, _max_polls=1)

        mock_run_repo.assert_not_called()


def test_watch_run_stops_cleanly_on_keyboard_interrupt():
    with tempfile.TemporaryDirectory() as root:
        _touch(os.path.join(root, "a.py"))
        with patch("main.run_repo"), \
             patch("main.time.sleep", side_effect=KeyboardInterrupt):
            watch_run(root, open_pr=False, max_iterations=5, initial_pass=False, _max_polls=None)
        # No exception propagated — that's the assertion.
