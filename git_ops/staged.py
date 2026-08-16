"""Git-staged-file discovery for --staged mode — process only files
currently staged for commit (`git add`), instead of every modernizable
file under a directory (repo mode) or a long-running poll loop (--watch).
This is the event-driven, pre-commit-hook-shaped complement to --watch:
triggered by the commit itself (fast — only touches what's actually
about to be committed), not a background daemon polling on a timer.
Typical use: a `.git/hooks/pre-commit` that runs `code-modernizer
--staged .` before every commit."""
import os
import subprocess


def get_staged_files(root_dir: str) -> list[str]:
    """Absolute paths of files staged for commit (`git diff --cached`),
    filtered to ones this project can actually chunk (has a registered
    language handler — see languages/__init__.py:get_handler). Only
    Added/Copied/Modified files (--diff-filter=ACM) — a staged DELETION
    has no content left on disk to modernize, and would otherwise show
    up here as a phantom "found" file that immediately fails to open.
    Returns [] (not an error) if root_dir isn't a git repository or
    nothing is staged — --staged mode simply has nothing to do then,
    not a failure worth raising over."""
    from languages import get_handler

    result = subprocess.run(
        ["git", "-C", root_dir, "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []

    files = []
    for rel_path in result.stdout.splitlines():
        if not rel_path.strip():
            continue
        full_path = os.path.join(root_dir, rel_path)
        if not os.path.isfile(full_path):
            continue
        try:
            get_handler(full_path)
        except ValueError:
            continue
        files.append(full_path)
    return sorted(files)
