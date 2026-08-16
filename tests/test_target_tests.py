import json
import os
import sys
import tempfile
from unittest.mock import patch

from target_tests import detect_test_command, run_test_command, run_target_tests_on_copy


def _write(root, rel_path, content):
    dest = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(content)


def test_detect_test_command_returns_none_for_empty_directory():
    with tempfile.TemporaryDirectory() as root:
        assert detect_test_command(root) is None


def test_detect_test_command_finds_pytest_via_pyproject():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "pyproject.toml", "[tool.pytest.ini_options]\n")
        label, command = detect_test_command(root)
        assert label == "pytest"
        assert "pytest" in command


def test_detect_test_command_finds_npm_with_real_test_script():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "package.json", json.dumps({"scripts": {"test": "jest"}}))
        label, command = detect_test_command(root)
        assert label == "npm"
        assert command == "npm test"


def test_detect_test_command_ignores_npm_placeholder_script():
    # `npm init`'s own default stub — a fresh, test-less JS project must
    # not look like it has a (guaranteed-failing) test suite.
    with tempfile.TemporaryDirectory() as root:
        _write(root, "package.json", json.dumps({
            "scripts": {"test": 'echo "Error: no test specified" && exit 1'}
        }))
        assert detect_test_command(root) is None


def test_detect_test_command_finds_maven():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "pom.xml", "<project></project>")
        label, command = detect_test_command(root)
        assert label == "maven"


def test_detect_test_command_finds_phpunit_config():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "phpunit.xml", "<phpunit></phpunit>")
        label, command = detect_test_command(root)
        assert label == "phpunit (config)"


def test_run_test_command_reports_success_on_zero_exit():
    with tempfile.TemporaryDirectory() as root:
        result = run_test_command(root, "exit 0")
        assert result["status"] == "success"
        assert result["exit_code"] == 0


def test_run_test_command_reports_failed_on_nonzero_exit():
    with tempfile.TemporaryDirectory() as root:
        result = run_test_command(root, "exit 1")
        assert result["status"] == "failed"
        assert result["exit_code"] == 1


def test_run_test_command_reports_timeout():
    with tempfile.TemporaryDirectory() as root:
        result = run_test_command(root, "sleep 5", timeout=1)
        assert result["status"] == "timeout"


def test_run_test_command_captures_stdout():
    with tempfile.TemporaryDirectory() as root:
        result = run_test_command(root, "echo hello")
        assert "hello" in result["stdout"]


def test_run_target_tests_on_copy_returns_none_when_no_framework_detected():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "main.py", "print('hi')\n")
        assert run_target_tests_on_copy(root, {}) is None


def _pytest_command():
    # detect_test_command hardcodes "python3 -m pytest -q" — the right
    # choice for the REAL feature (a target repo's own activated venv is
    # expected to be on PATH when this runs on the host, the same way a
    # user would invoke it themselves). But bare "python3" on THIS dev
    # machine resolves to the system interpreter, not this project's own
    # .venv, so it doesn't have pytest importable — these tests instead
    # pin to sys.executable (guaranteed to have pytest, since it's what's
    # currently running this very test) to test the copy/overlay
    # mechanism itself, independent of that unrelated PATH concern.
    return f"{sys.executable} -m pytest -q"


def test_run_target_tests_on_copy_runs_against_a_copy_not_the_original():
    # The critical safety property: the ORIGINAL directory must be
    # byte-for-byte unchanged after the call, even though the overlay
    # writes new content — that content must only ever land in the
    # temporary copy, never the user's real working tree.
    with tempfile.TemporaryDirectory() as root:
        _write(root, "pyproject.toml", "[tool.pytest.ini_options]\n")
        _write(root, "calc.py", "def add(a, b):\n    return a + b\n")
        _write(root, "test_calc.py", "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")

        with patch("target_tests.detect_test_command", return_value=("pytest", _pytest_command())):
            result = run_target_tests_on_copy(root, {"calc.py": "def add(a, b):\n    return a - b\n"})

        # Original file on disk must be untouched.
        with open(os.path.join(root, "calc.py")) as f:
            assert f.read() == "def add(a, b):\n    return a + b\n"

        # But the test run (against the mutated COPY) must reflect the
        # overlay — the broken "a - b" version should fail add(2,3)==5.
        assert result is not None
        assert result["status"] == "failed", result["stderr"]


def test_run_target_tests_on_copy_passes_when_overlay_keeps_tests_green():
    with tempfile.TemporaryDirectory() as root:
        _write(root, "pyproject.toml", "[tool.pytest.ini_options]\n")
        _write(root, "calc.py", "def add(a, b):\n    return a + b\n")
        _write(root, "test_calc.py", "from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n")

        # Overlay with an EQUIVALENT (not broken) rewrite.
        with patch("target_tests.detect_test_command", return_value=("pytest", _pytest_command())):
            result = run_target_tests_on_copy(root, {"calc.py": "def add(a, b):\n    return (a + b)\n"})

        assert result is not None
        assert result["status"] == "success", result["stderr"]


def test_run_target_tests_on_copy_symlinks_node_modules_instead_of_excluding_it():
    # node_modules must be reachable in the copy or `npm test` fails on
    # missing deps regardless of whether the modernization is correct —
    # that false "regression" would wrongly block --pr. Prove the
    # installed-dependency dir survives into the copy as a symlink back
    # to the original (not copied wholesale, not silently dropped).
    with tempfile.TemporaryDirectory() as root:
        _write(root, "package.json", json.dumps({"scripts": {"test": "true"}}))
        node_modules = os.path.join(root, "node_modules")
        os.makedirs(node_modules)
        _write(root, "node_modules/some-dep/index.js", "module.exports = {};\n")

        captured = {}
        real_run = run_test_command

        def _spy(tmp_copy, command, timeout=120):
            # Must assert HERE, not after run_target_tests_on_copy
            # returns — it deletes the temp copy in its `finally` block,
            # so the path won't exist anymore by then.
            node_modules_path = os.path.join(tmp_copy, "node_modules")
            captured["is_link"] = os.path.islink(node_modules_path)
            captured["realpath"] = os.path.realpath(node_modules_path)
            return real_run(tmp_copy, command, timeout=timeout)

        with patch("target_tests.run_test_command", side_effect=_spy):
            result = run_target_tests_on_copy(root, {})

        assert result is not None
        assert captured["is_link"] is True
        assert captured["realpath"] == os.path.realpath(node_modules)
