"""Git-worktree isolation for concurrent repo-mode runs (--workers > 1
combined with --isolate-workers).

Worth being explicit about what this DOES and DOESN'T fix, since it's
easy to oversell: run_file() already only READS the target file once at
the start (never mutates it) and only ever WRITES to a sibling
`.modernized.<ext>` path unique to that file — two workers processing
DIFFERENT files never touch the same path, so there is no existing race
condition between concurrent workers in the current pipeline. Sandbox
compile/run happens inside a throwaway Docker temp dir per verify() call
(sandbox/verifier.py), not on the host, so it can't collide with the
host checkout either.

What this DOES add: each worker's file read (and, if a future feature
ever adds host-side writes mid-run — a linter, a formatter, a language
server touching the checkout) happens against its OWN full checkout of
the repo at HEAD, isolated from every other worker AND from a human
concurrently editing the same working tree while a run is in progress.
That's real defense-in-depth for exactly the failure mode the "run
parallel AI agents in git worktrees" pattern targets, even though
nothing in this pipeline exploits it yet — worth having before a future
feature needs it, not after a run corrupts something because two workers
raced on a file that used to be safe to share.

Opt-in (--isolate-workers, default off) because it isn't free: every
worktree is a full `git checkout` (fast — a worktree shares the .git
object store, no new clone — but not zero-cost) plus copying each
successful chunk's output back into the real tree afterward."""
import os
import shutil
import subprocess
import tempfile


class NotAGitRepoError(Exception):
    """root_dir isn't a git repository — --isolate-workers has nothing
    to check a worktree out FROM. Callers should fail loudly rather than
    silently falling back to unisolated mode, since the user explicitly
    opted in."""


def is_git_repo(root_dir: str) -> bool:
    result = subprocess.run(
        ["git", "-C", root_dir, "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def create_worktree(root_dir: str) -> str:
    """Check out a new, detached worktree of root_dir's current HEAD into
    a fresh temp directory and return its path. Detached (not a new
    branch) because this worktree is read/write scratch space for ONE
    file's modernization run, never something meant to be committed to
    or pushed from directly — PR creation happens entirely through the
    GitHub API against the real repo (git_ops/pr.py), never through this
    worktree's local git state."""
    if not is_git_repo(root_dir):
        raise NotAGitRepoError(
            f"--isolate-workers requires root_dir to be a git repository, but "
            f"{root_dir!r} isn't one (no .git found via `git rev-parse`)."
        )
    parent = tempfile.mkdtemp(prefix="code-modernizer-worktree-")
    worktree_path = os.path.join(parent, "wt")
    result = subprocess.run(
        ["git", "-C", root_dir, "worktree", "add", "--detach", worktree_path, "HEAD"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(parent, ignore_errors=True)
        raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    return worktree_path


def remove_worktree(root_dir: str, worktree_path: str) -> None:
    """Best-effort cleanup — a failed removal must never crash a run that
    otherwise completed successfully. Tries `git worktree remove` first
    (the correct way: also drops the worktree's registration from
    root_dir/.git/worktrees, so `git worktree list` stays accurate);
    falls back to a plain rmtree + `git worktree prune` if that fails
    (e.g. the worktree already has uncommitted changes git refuses to
    discard without --force, though --force is already passed first)."""
    result = subprocess.run(
        ["git", "-C", root_dir, "worktree", "remove", "--force", worktree_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        subprocess.run(["git", "-C", root_dir, "worktree", "prune"], capture_output=True, text=True)
    # Remove the temp parent dir (worktree_path's grandparent from
    # create_worktree) regardless of which cleanup path ran above.
    shutil.rmtree(os.path.dirname(worktree_path), ignore_errors=True)
