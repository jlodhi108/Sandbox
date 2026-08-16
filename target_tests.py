"""Detect and run the TARGET repo's own pre-existing test suite (pytest,
npm test, PHPUnit, Maven/Gradle) as an additional, independent oracle —
separate from and complementary to this project's own generated
baseline/probe checks.

This is DELIBERATELY different from everything else in this codebase:
sandbox/verifier.py runs code in an isolated, network-disabled Docker
container because it's executing LLM-GENERATED snippets it cannot trust.
This module instead runs the target repo's OWN human-authored test
command directly on the HOST, because (a) it needs that repo's actual
installed dependencies (venv, node_modules, vendor/) which only exist on
the host, and the sandbox has no network access or install step to
acquire them (confirmed empirically — see agents/nodes.py's REQUIRES-
resolvability check for the full story); (b) the test command itself is
pre-existing code the user already trusts and already runs on their own
machine as part of their normal workflow, not something this project
generated. That's a genuinely different trust boundary from the rest of
this project, which is exactly why every caller of run_test_command
MUST be opt-in (never default-on) and why callers apply it to a
TEMPORARY COPY of the repo (see main.py's _run_target_tests_after),
never the user's real working tree.
"""
import os
import shutil
import subprocess

# (glob-free) file/dir existence checks -> (framework label, shell command).
# Order matters: checked top to bottom, first match wins. A repo could
# plausibly match more than one (e.g. both package.json and pom.xml in
# a monorepo) — this picks the first one found rather than trying to
# run every framework present, keeping behavior predictable.
_DETECTORS = (
    ("pytest", ("pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py"), "python3 -m pytest -q"),
    ("npm", ("package.json",), None),  # special-cased below: needs a real "test" script, not just the file
    ("phpunit (vendor)", ("vendor/bin/phpunit",), "vendor/bin/phpunit"),
    ("phpunit (config)", ("phpunit.xml", "phpunit.xml.dist"), "vendor/bin/phpunit"),
    ("maven", ("pom.xml",), "mvn -q -B test"),
    ("gradle (wrapper)", ("gradlew",), "./gradlew test -q"),
    ("gradle", ("build.gradle", "build.gradle.kts"), "gradle test -q"),
)


def _npm_test_command(root_dir: str) -> str | None:
    """package.json existing isn't enough — `npm init`'s own default
    stub script is `"echo \\"Error: no test specified\\" && exit 1"`,
    which would make every fresh JS project look like a failing test
    suite. Only treat it as a real test command if the script differs
    from that exact placeholder."""
    import json
    try:
        with open(os.path.join(root_dir, "package.json")) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    script = data.get("scripts", {}).get("test", "")
    if not script or "Error: no test specified" in script:
        return None
    return "npm test"


def detect_test_command(root_dir: str) -> tuple[str, str] | None:
    """Returns (framework_label, shell_command) for the first recognized
    test setup found at the root of root_dir, or None if nothing
    matched. Deliberately shallow (root-level files only, no recursive
    search) — matches how each of these tools' own convention works
    (pytest/npm/maven/gradle/phpunit are all invoked from a project
    root, not an arbitrary subdirectory) and keeps detection fast and
    predictable rather than guessing at nested project layouts."""
    for label, markers, command in _DETECTORS:
        if not any(os.path.exists(os.path.join(root_dir, m)) for m in markers):
            continue
        if label == "npm":
            npm_command = _npm_test_command(root_dir)
            if npm_command is None:
                continue
            return label, npm_command
        return label, command
    return None


def run_test_command(root_dir: str, command: str, timeout: int = 120) -> dict:
    """Runs `command` with cwd=root_dir directly on the HOST — see this
    module's docstring for why. Returns {"status": "success"|"failed"|
    "timeout"|"error", "stdout", "stderr", "exit_code"}. "failed" means
    the command ran and reported a nonzero exit code (tests ran, some
    failed) — a normal, expected outcome to compare before/after, not a
    tooling problem. "error"/"timeout" mean the command itself couldn't
    be evaluated at all (framework not actually installed, hung past
    timeout, etc.) — callers should treat those as "no usable result,"
    not as "tests failed.\""""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=root_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "success" if result.returncode == 0 else "failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stdout": "", "stderr": f"Timed out after {timeout}s", "exit_code": None}
    except OSError as e:
        return {"status": "error", "stdout": "", "stderr": str(e), "exit_code": None}


def run_target_tests_on_copy(root_dir: str, overlay_files: dict[str, str], timeout: int = 120) -> dict | None:
    """Copies root_dir to a temp directory, overlays overlay_files
    ({relative_path: new_content}) on top of that COPY, detects and runs
    the repo's test command IN THE COPY, then deletes the copy — the
    user's real working tree is never touched. Returns run_test_command's
    result dict, or None if no test command was detected. This is the
    only entry point that should ever be used to test MODERNIZED content
    against a repo's test suite — testing in place would mean running
    the repo's own setup/teardown/fixtures against a real working tree
    for code this project generated, which is exactly the kind of
    irreversible risk this project avoids everywhere else via sandboxing."""
    detected = detect_test_command(root_dir)
    if detected is None:
        return None
    _, command = detected

    import tempfile
    tmp_parent = tempfile.mkdtemp()
    try:
        tmp_copy = os.path.join(tmp_parent, "repo")
        # node_modules/.venv/venv are excluded from the copy itself (they
        # can be huge and copying them is slow), but the test command
        # still needs them to be ABLE to run — without this, `npm test`
        # or a venv-relative pytest invocation fails on missing deps in
        # the copy every time, which looks exactly like a modernization
        # regression (before: passes on the real repo; after: "error" on
        # the copy) and would wrongly block --pr. Symlinking them in is
        # safe: nothing under overlay_files ever targets these dirs (they
        # aren't source files this project modernizes), and the copy
        # itself is a throwaway temp dir deleted in `finally` below.
        install_dirs = ("node_modules", ".venv", "venv")
        shutil.copytree(
            root_dir, tmp_copy,
            ignore=shutil.ignore_patterns(".git", "__pycache__", *install_dirs),
        )
        for install_dir in install_dirs:
            src = os.path.join(root_dir, install_dir)
            if os.path.isdir(src):
                os.symlink(src, os.path.join(tmp_copy, install_dir))
        for rel_path, content in overlay_files.items():
            dest = os.path.join(tmp_copy, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w") as f:
                f.write(content)
        return run_test_command(tmp_copy, command, timeout=timeout)
    finally:
        shutil.rmtree(tmp_parent, ignore_errors=True)
