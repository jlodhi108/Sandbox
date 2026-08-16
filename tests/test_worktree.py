import os
import subprocess
import tempfile

from git_ops.worktree import is_git_repo, create_worktree, remove_worktree, NotAGitRepoError


def _init_repo(root: str) -> None:
    subprocess.run(["git", "init", "-q", root], check=True)
    subprocess.run(["git", "-C", root, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", root, "config", "user.name", "Test"], check=True)
    with open(os.path.join(root, "calc.py"), "w") as f:
        f.write("def add(a, b):\n    return a + b\n")
    subprocess.run(["git", "-C", root, "add", "."], check=True)
    subprocess.run(["git", "-C", root, "commit", "-q", "-m", "initial"], check=True)


def test_is_git_repo_true_for_a_real_repo():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        assert is_git_repo(root) is True


def test_is_git_repo_false_for_a_plain_directory():
    with tempfile.TemporaryDirectory() as root:
        assert is_git_repo(root) is False


def test_create_worktree_raises_for_non_git_directory():
    with tempfile.TemporaryDirectory() as root:
        try:
            create_worktree(root)
            assert False, "expected NotAGitRepoError"
        except NotAGitRepoError:
            pass


def test_create_worktree_checks_out_a_working_copy_at_head():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        worktree_path = create_worktree(root)
        try:
            assert os.path.isfile(os.path.join(worktree_path, "calc.py"))
            with open(os.path.join(worktree_path, "calc.py")) as f:
                assert "def add" in f.read()
        finally:
            remove_worktree(root, worktree_path)


def test_worktree_is_isolated_from_the_original():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        worktree_path = create_worktree(root)
        try:
            # Writing inside the worktree must NOT touch root_dir's copy.
            with open(os.path.join(worktree_path, "calc.py"), "w") as f:
                f.write("def add(a, b):\n    return a + b  # edited in worktree\n")
            with open(os.path.join(root, "calc.py")) as f:
                assert "edited in worktree" not in f.read()
        finally:
            remove_worktree(root, worktree_path)


def test_remove_worktree_cleans_up_registration_and_files():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        worktree_path = create_worktree(root)
        remove_worktree(root, worktree_path)

        assert not os.path.exists(worktree_path)
        listing = subprocess.run(
            ["git", "-C", root, "worktree", "list"], capture_output=True, text=True,
        ).stdout
        assert worktree_path not in listing


def test_multiple_worktrees_are_independent():
    with tempfile.TemporaryDirectory() as root:
        _init_repo(root)
        wt1 = create_worktree(root)
        wt2 = create_worktree(root)
        try:
            assert wt1 != wt2
            with open(os.path.join(wt1, "calc.py"), "w") as f:
                f.write("# only in wt1\n")
            with open(os.path.join(wt2, "calc.py")) as f:
                assert "only in wt1" not in f.read()
        finally:
            remove_worktree(root, wt1)
            remove_worktree(root, wt2)
