import docker
import json
import tempfile
import shutil
import os
import time
import requests.exceptions

_client = None

# Transient Docker-daemon hiccups (a momentary API disconnect, the daemon
# briefly restarting) are the same class of flakiness
# agents/nodes.py:_invoke_llm_with_retry already retries for Ollama —
# without this, one blip hard-fails a chunk's verification outright and
# wastes an otherwise-good LLM candidate. Deliberately small: this is a
# local daemon on the same machine, not a network service, so a real
# outage should surface quickly rather than being masked by long retries.
_DOCKER_RETRY_ATTEMPTS = 3
_DOCKER_RETRY_DELAY_SECONDS = 2


def _run_container_with_retry(client, **run_kwargs):
    last_error = None
    for attempt in range(_DOCKER_RETRY_ATTEMPTS):
        try:
            return client.containers.run(**run_kwargs)
        except (docker.errors.APIError, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < _DOCKER_RETRY_ATTEMPTS - 1:
                time.sleep(_DOCKER_RETRY_DELAY_SECONDS)
    raise ConnectionError(
        f"Could not start sandbox container after {_DOCKER_RETRY_ATTEMPTS} attempts — "
        f"is the Docker daemon running? Original error: {last_error}"
    ) from last_error

# Optional stronger container isolation (e.g. gVisor's "runsc") for the
# containers that actually EXECUTE untrusted code. Off by default — a
# normal Docker install has no such runtime registered, and passing one
# that doesn't exist is a hard Docker error, so this must never activate
# unless explicitly configured. This is a genuinely host-level, Linux-
# only setup step (installing gVisor and registering it with the Docker
# daemon) that this project can wire up but cannot itself provide —
# native Docker on Linux (a CI runner, an EC2 box) is where this
# actually installs; Docker Desktop's managed VM (macOS/Windows) has no
# supported way to add a runtime to it at all.
SANDBOX_RUNTIME = os.environ.get("SANDBOX_RUNTIME") or None

# Custom seccomp profile, ON by default — unlike SANDBOX_RUNTIME, this
# needs NO host-level setup (it's a plain JSON file shipped in this repo)
# and works identically on Linux and macOS Docker Desktop, so it closes
# the "gVisor is a Linux-only no-op" hardening gap for local dev without
# asking anyone to configure anything. Derived from Docker's own
# default seccomp profile (moby/profiles, fetched 2026-08) with ONE
# change: the unconditional allow for ptrace/process_vm_readv/
# process_vm_writev (gated only by kernel version, not by capability,
# in Docker's stock profile) is removed entirely, falling through to
# the profile's defaultAction (SCMP_ACT_ERRNO — deny). This sandbox
# only ever compiles/runs small, known programs across 6 language
# toolchains — none of them have any legitimate reason to inspect or
# attach to another process's memory, which is exactly what those 3
# syscalls are for and exactly the kind of capability a container-escape
# or info-leak technique would want. Verified against all 6 toolchains
# (g++, python3, node, tsc, javac/java, php) before being trusted here —
# see sandbox/verifier.py's __main__ self-test, which both the CI
# "sandbox-smoke-test" job and this comment's author ran directly.
# Everything else Docker's default profile already allows/blocks is left
# completely untouched, deliberately: that profile is mature and
# carefully tuned (e.g. per-argument filtering on `personality`,
# capability-gating an entire second tier of syscalls like mount/
# unshare/setns), and hand-rewriting more of it from scratch risks
# subtly breaking one of the 6 toolchains in a way that would look like
# a confusing compiler bug rather than a clear security-config error.
#
# Set DISABLE_SECCOMP_PROFILE=1 to fall back to Docker's own unmodified
# default profile (e.g. if a future toolchain genuinely needs one of the
# 3 removed syscalls — unlikely, but this is the escape hatch).
_SECCOMP_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seccomp-profile.json")
_seccomp_profile_json = None
if not os.environ.get("DISABLE_SECCOMP_PROFILE"):
    with open(_SECCOMP_PROFILE_PATH) as _f:
        _seccomp_profile_json = _f.read()


def _get_client():
    # Lazy so importing this module doesn't require a running Docker
    # daemon — only calling verify() does.
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


def _security_opt() -> list[str] | None:
    if _seccomp_profile_json is None:
        return None
    return [f"seccomp={_seccomp_profile_json}"]


def verify(
    source_code: str,
    filename: str,
    run_cmd: str,
    timeout: int = 15,
    image: str = "sandbox-multi",
) -> dict:
    """Write `source_code` to `filename` inside an isolated container and
    run `run_cmd` (a shell command expected to both check correctness —
    compile/lint — and execute, so runtime failures are caught too, not
    just syntax errors)."""
    client = _get_client()

    # Not using tempfile.TemporaryDirectory()'s context manager: its
    # cleanup() walks the tree and chmods/unlinks everything strictly,
    # which raises if any file is owned by a uid it doesn't own. The
    # container writes compiler output (a.out, .class files, tsc's
    # --outDir) as its own non-root user (uid 1000) — on native Linux
    # Docker (unlike macOS Docker Desktop's VM layer, which papers over
    # this) those artifacts are genuinely owned by uid 1000 on the host
    # filesystem, and the CI runner's user can neither chmod nor delete
    # something it doesn't own and isn't root for. ignore_errors=True
    # leaves those specific orphaned files behind (harmless on ephemeral
    # CI runners; worth a periodic /tmp sweep on a long-lived box) instead
    # of crashing every verify() call that happens to produce output files.
    tmpdir = tempfile.mkdtemp()
    try:
        os.chmod(tmpdir, 0o777)

        src_path = os.path.join(tmpdir, filename)
        with open(src_path, "w") as f:
            f.write(source_code)
        os.chmod(src_path, 0o644)

        container = None
        try:
            run_kwargs = dict(
                image=image,
                command=["bash", "-c", run_cmd],
                volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                mem_limit="512m",
                nano_cpus=1_000_000_000,
                network_disabled=True,
                user="sandboxuser",
                detach=True,
            )
            if SANDBOX_RUNTIME:
                run_kwargs["runtime"] = SANDBOX_RUNTIME
            security_opt = _security_opt()
            if security_opt:
                run_kwargs["security_opt"] = security_opt
            container = _run_container_with_retry(client, **run_kwargs)

            try:
                result = container.wait(timeout=timeout)
                # Combined stdout+stderr for error feedback to the model —
                # compiler errors land on either stream depending on the
                # tool, so mixing them is what we want there. `stdout_only`
                # is separate and used for behavioral-equivalence checks,
                # where mixing in stderr noise (e.g. -Wall warnings that
                # can legitimately differ between two working versions)
                # would cause false-positive "output changed" mismatches.
                combined_logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                stdout_only = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
                exit_code = result["StatusCode"]
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                # container.wait's OWN timeout param raises exactly this —
                # the container is still running past `timeout` seconds,
                # a real hang (e.g. an infinite loop in the candidate
                # code), not a Docker problem.
                container.kill()
                return {"status": "failed", "stderr": "Execution timed out", "stdout": "", "exit_code": -1}
            except docker.errors.APIError as e:
                # A genuine daemon-side error (API disconnect, container
                # OOM-killed, daemon restart mid-run) — mislabeling this
                # as "Execution timed out" is misleading in reports AND
                # to the fix-retry loop, which would otherwise retry a
                # compile that never actually ran.
                container.kill()
                return {"status": "failed", "stderr": f"Docker error: {e}", "stdout": "", "exit_code": -1}

            return {
                "status": "success" if exit_code == 0 else "failed",
                "stderr": combined_logs,
                "stdout": stdout_only,
                "exit_code": exit_code,
            }
        finally:
            if container is not None:
                container.remove(force=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_semgrep(source_code: str, filename: str, timeout: int = 15, image: str = "sandbox-multi") -> dict:
    """Static-analysis security scan on `source_code` — does NOT execute
    it, so it needs the same container isolation as verify() only for
    consistency, not because scanning itself is risky. Uses a LOCAL,
    offline rule file (sandbox/security-rules.yaml, baked into the image
    at /opt/security-rules.yaml) rather than a semgrep registry config
    (`p/...`): registry configs try to revalidate over the network on
    every run even when previously cached, and inside a
    network_disabled container that doesn't fail fast — confirmed by
    testing directly, it took ~97 seconds to eventually fall back.
    --metrics=off and --disable-version-check are BOTH required to avoid
    two separate slow network calls; neither flag alone is enough."""
    client = _get_client()
    tmpdir = tempfile.mkdtemp()
    try:
        os.chmod(tmpdir, 0o777)
        src_path = os.path.join(tmpdir, filename)
        with open(src_path, "w") as f:
            f.write(source_code)
        os.chmod(src_path, 0o644)

        cmd = (
            f"semgrep --config=/opt/security-rules.yaml --metrics=off "
            f"--disable-version-check --json {filename}"
        )
        stdout = ""
        container = None
        try:
            run_kwargs = dict(
                image=image,
                command=["bash", "-c", cmd],
                volumes={tmpdir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                mem_limit="512m",
                nano_cpus=1_000_000_000,
                network_disabled=True,
                user="sandboxuser",
                detach=True,
            )
            if SANDBOX_RUNTIME:
                run_kwargs["runtime"] = SANDBOX_RUNTIME
            security_opt = _security_opt()
            if security_opt:
                run_kwargs["security_opt"] = security_opt
            container = _run_container_with_retry(client, **run_kwargs)
            try:
                container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            except Exception:
                container.kill()
                return {"status": "error", "findings": []}
        finally:
            if container is not None:
                container.remove(force=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return {"status": "error", "findings": []}

    findings = [
        {
            "rule_id": r["check_id"],
            "line": r["start"]["line"],
            "message": r["extra"]["message"],
        }
        for r in data.get("results", [])
    ]
    return {"status": "success", "findings": findings}


if __name__ == "__main__":
    import sys

    checks = [
        ("main.cpp", '#include <iostream>\nint main() { std::cout << "cpp ok\\n"; return 0; }\n',
         "g++ -std=c++20 -Wall main.cpp -o main && ./main"),
        ("main.py", 'print("python ok")\n', "python3 main.py"),
        ("main.js", 'console.log("js ok");\n', "node main.js"),
        ("main.ts", 'const x: number = 1;\nconsole.log("ts ok", x);\n',
         "npx tsc main.ts --target ES2020 --module commonjs --outDir out --skipLibCheck && node out/main.js"),
        ("Main.java", 'public class Main { public static void main(String[] a) { System.out.println("java ok"); } }\n',
         "javac Main.java && java Main"),
        ("main.php", '<?php\necho "php ok\\n";\n', "php -l main.php && php main.php"),
    ]

    failures = []
    for filename, code, cmd in checks:
        result = verify(code, filename, cmd)
        print(f"{filename}: {result}")
        if result["status"] != "success":
            failures.append(filename)

    if failures:
        print(f"FAILED toolchains: {failures}")
        sys.exit(1)
    print("All toolchains OK")

    if _seccomp_profile_json is not None:
        # Positive confirmation the hardened profile is actually ACTIVE
        # (not just that nothing broke) — ptrace must now be blocked,
        # where Docker's stock default profile allows it unconditionally.
        ptrace_probe = (
            "import ctypes, ctypes.util\n"
            "libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)\n"
            "result = libc.ptrace(0, 0, None, None)\n"
            "print('BLOCKED' if result == -1 and ctypes.get_errno() != 0 else 'ALLOWED')\n"
        )
        ptrace_result = verify(ptrace_probe, "main.py", "python3 main.py")
        print(f"ptrace probe: {ptrace_result}")
        if "BLOCKED" not in ptrace_result.get("stdout", ""):
            print("FAILED: seccomp profile did not block ptrace as expected")
            sys.exit(1)
        print("Seccomp hardening OK (ptrace blocked)")

    semgrep_result = run_semgrep('import os\ndef f(cmd):\n    os.system(cmd)\n', "main.py")
    print(f"semgrep: {semgrep_result}")
    if semgrep_result["status"] != "success" or not semgrep_result["findings"]:
        print("FAILED: semgrep did not detect the known-vulnerable sample")
        sys.exit(1)
    print("Semgrep security gate OK")
