import os
import subprocess
import tempfile

from git_ops.staged import get_staged_files


def _init_repo(root: str) -> None:
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "-C", root, "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "T"], check=True)


def _write(root: str, rel_path: str, content: str) -> None:
    dest = os.path.join(root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(content)


def test_returns_staged_modernizable_files():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        subprocess.run(["git", "-C", root, "add", "a.py"], check=True)

        assert get_staged_files(root) == [os.path.join(root, "a.py")]


def test_excludes_files_without_a_language_handler():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        _write(root, "readme.txt", "not code\n")
        subprocess.run(["git", "-C", root, "add", "a.py", "readme.txt"], check=True)

        assert get_staged_files(root) == [os.path.join(root, "a.py")]


def test_excludes_unstaged_files():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        subprocess.run(["git", "-C", root, "add", "a.py"], check=True)
        _write(root, "b.py", "def g(x):\n    return x\n")  # never git add'd

        assert get_staged_files(root) == [os.path.join(root, "a.py")]


def test_excludes_files_only_modified_after_staging():
    # git diff --cached reflects the STAGED snapshot, not the working
    # tree — a file staged then edited again should still be picked up
    # (it's still staged, just with older content than what's on disk
    # now — git diff --name-only --cached lists it regardless).
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        subprocess.run(["git", "-C", root, "add", "a.py"], check=True)
        _write(root, "a.py", "def f(x):\n    return x + 1\n")  # edited after staging

        assert get_staged_files(root) == [os.path.join(root, "a.py")]


def test_excludes_staged_deletions():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        subprocess.run(["git", "-C", root, "add", "a.py"], check=True)
        subprocess.run(["git", "-C", root, "commit", "-q", "-m", "initial"], check=True)
        os.remove(os.path.join(root, "a.py"))
        subprocess.run(["git", "-C", root, "add", "a.py"], check=True)  # stages the deletion

        assert get_staged_files(root) == []


def test_returns_empty_list_for_non_git_directory():
    with tempfile.TemporaryDirectory() as root:
        assert get_staged_files(root) == []


def test_returns_empty_list_when_nothing_staged():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        _write(root, "a.py", "def f(x):\n    return x\n")
        assert get_staged_files(root) == []
