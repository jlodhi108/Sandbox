import docker
import tempfile
import shutil
import os

_client = None


def _get_client():
    # Lazy so importing this module doesn't require a running Docker
    # daemon — only calling verify() does.
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


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
            container = client.containers.run(
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

            try:
                result = container.wait(timeout=timeout)
                logs = container.logs().decode("utf-8", errors="replace")
                exit_code = result["StatusCode"]
            except Exception:
                container.kill()
                return {"status": "failed", "stderr": "Execution timed out", "exit_code": -1}

            return {
                "status": "success" if exit_code == 0 else "failed",
                "stderr": logs,
                "exit_code": exit_code,
            }
        finally:
            if container is not None:
                container.remove(force=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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
