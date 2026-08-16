import argparse
import difflib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from config import load_config, apply_config_to_environment

# Must run before importing agents.graph / sandbox.verifier — both read
# environment variables (ESCALATION_MODEL, SANDBOX_RUNTIME) at import
# time, so anything configured only in .env or .modernizer.toml (not
# already exported in the shell) would silently never take effect if
# this ran after those imports. Precedence: shell env > .env >
# .modernizer.toml > hardcoded default — load_dotenv() first, then
# apply_config_to_environment()'s setdefault() calls only fill in what's
# still unset.
load_dotenv()
_config = load_config()
apply_config_to_environment(_config)

from languages import get_handler
from agents.graph import modernize
from sandbox.verifier import verify
from git_ops.pr import open_modernization_pr, open_multi_file_pr
from track_record import load_history, save_history, record_run, is_eligible

# Loaded once at startup, read against for every auto-PR eligibility
# check this run makes — always reflects PRIOR runs only. Updated with
# THIS run's results and persisted at the very end (see __main__), never
# mid-run, so a run can't bootstrap its own eligibility using its own
# in-progress results.
_history = load_history()
_autonomy_settings = _config.get("autonomy", {})
_MIN_TRACK_RECORD_CHUNKS = _autonomy_settings.get("min_track_record_chunks", 5)
_MIN_SUCCESS_RATE = _autonomy_settings.get("min_success_rate", 0.8)

# Common vendor/build/VCS directories to never descend into in repo mode —
# nothing in here is source you'd want modernized, and node_modules/.git
# in particular can be enormous.
_EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", "out", ".next", "target", ".pytest_cache",
}


def _unified_diff(original: str, modernized: str, file_path: str) -> str:
    """Per-chunk diff, not whole-file: GitHub already renders a full-file
    diff natively in the PR's own "Files changed" view once we commit the
    new content, so duplicating that into the PR body text would just be
    redundant. What's actually missing is any record of what changed in
    the JSON report — right now it only has pass/fail stats, nothing
    showing the actual edit — so that's what this fills in."""
    # Chunk text comes from byte-slicing (see CodeChunk) and never has a
    # trailing newline. Without one, difflib's last "-"/"+" line pair
    # prints with no line break between them (glued together like
    # "-}+};") since neither line contributes its own separator. Adding
    # it back here is purely cosmetic — doesn't touch the actual
    # modernized_code used for splicing anywhere else.
    if original and not original.endswith("\n"):
        original += "\n"
    if modernized and not modernized.endswith("\n"):
        modernized += "\n"

    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        modernized.splitlines(keepends=True),
        fromfile=f"{file_path} (before)",
        tofile=f"{file_path} (after)",
    )
    return "".join(diff_lines)


def _format_flags_section(risk_flagged: list[dict], security_flagged: list[dict]) -> str:
    """Shared by the single-file PR body and (via the same shape) the
    combined repo-mode PR body — two independent flag types, surfaced
    separately since they catch different things: risk_flag is "this
    touches something a stdout-diff can't fully verify" (LLM
    self-critique), security_flag is "static analysis found a specific
    pattern" (semgrep). A chunk can be flagged by either, both, or
    neither."""
    parts = []
    if risk_flagged:
        parts.append(
            f"**{len(risk_flagged)} chunk(s) flagged for manual review** "
            f"(touch I/O, global state, randomness, or timing — the "
            f"automated stdout check can't fully prove these are safe):\n"
            + "\n".join(f"- {r['kind']} at byte {r['start_byte']}: {r['reason']}" for r in risk_flagged)
        )
    else:
        parts.append("No chunks flagged as higher-risk.")

    if security_flagged:
        finding_lines = [
            f"- {s['kind']} at byte {s['start_byte']}: {f['rule_id']} — {f['message']}"
            for s in security_flagged for f in s["findings"]
        ]
        parts.append(
            f"**{len(security_flagged)} chunk(s) flagged by static security scan** "
            f"(semgrep, local rules — see sandbox/security-rules.yaml):\n"
            + "\n".join(finding_lines)
        )
    else:
        parts.append("No security findings.")

    return "\n\n".join(parts)


def run_file(
    file_path: str, open_pr: bool, max_iterations: int,
    standalone_pr: bool = True, sibling_sources: list[bytes] | None = None,
) -> dict:
    """Modernize one file. Returns a stats dict (also the data source for
    --report) rather than just printing, so repo mode can aggregate
    results across many files without re-parsing terminal output.

    standalone_pr controls whether THIS call opens its own single-file
    PR when open_pr is set. Repo mode passes False here and instead
    collects every file's output itself, opening ONE combined multi-file
    PR at the end — GitHub has no atomic multi-file update via the
    single-file convenience API, so letting each file open its own PR
    would mean N separate PRs for one repo-wide run, which is exactly
    the noisy outcome multi-file mode exists to avoid.

    sibling_sources: raw bytes of every OTHER file in the repo (repo mode
    only — empty/None for standalone single-file runs), searched for real
    calls to each function being modernized so probes can use actual
    usage instead of an LLM-guessed example."""
    start_time = time.time()
    handler = get_handler(file_path)
    with open(file_path, "rb") as f:
        source = f.read()

    chunks = handler.chunk(source)
    print(f"[{handler.name}] Found {len(chunks)} chunk(s) to modernize in {file_path}")

    stats = {
        "file_path": file_path,
        "language": handler.name,
        "chunks_total": len(chunks),
        "chunks_succeeded": 0,
        "chunks_already_modern": 0,
        "chunks_gave_up": 0,
        "risk_flagged": [],
        "security_flagged": [],
        "chunk_details": [],
        "output_path": None,
        "final_check_passed": None,
        "pr_url": None,
        "duration_seconds": None,
    }

    # Process bottom-to-top and accumulate into `working_source` so each
    # chunk is verified against everything already modernized before it
    # (not just the pristine original). Editing from the end of the file
    # backward keeps every not-yet-processed chunk's byte offsets valid,
    # since a splice never shifts bytes that come before it.
    working_source = source
    collected_imports: list[str] = []  # merged in once, AFTER the loop —
    # prepending mid-loop would shift byte offsets for chunks not yet
    # processed, since they all sit before the insertion point
    for chunk in sorted(chunks, key=lambda c: c.start_byte, reverse=True):
        print(f"\n--- Modernizing {chunk.kind} [{chunk.start_byte}:{chunk.end_byte}] ---")
        chunk_start_time = time.time()

        if handler.already_modern(chunk.code):
            print("SKIPPED (already modern — no LLM call made)")
            stats["chunks_already_modern"] += 1
            stats["chunk_details"].append({
                "kind": chunk.kind, "start_byte": chunk.start_byte,
                "status": "already_modern", "iterations": 0,
                "duration_seconds": time.time() - chunk_start_time,
            })
            continue

        final_state = modernize(
            handler.name, working_source, chunk.start_byte, chunk.end_byte,
            max_iterations=max_iterations, sibling_sources=sibling_sources,
        )
        print(f"status={final_state['status']} iterations={final_state['iteration_count']}")

        chunk_detail = {
            "kind": chunk.kind,
            "start_byte": chunk.start_byte,
            "status": final_state["status"],
            "iterations": final_state["iteration_count"],
            "used_escalation": final_state.get("used_escalation", False),
            "probe_count": len(final_state.get("probes") or []),
            "risk_flag": final_state.get("risk_flag", False),
            "security_flag": final_state.get("security_flag", False),
            "duration_seconds": time.time() - chunk_start_time,
        }

        if final_state["status"] == "success":
            diff_text = _unified_diff(chunk.code, final_state["modernized_code"], file_path)
            chunk_detail["diff"] = diff_text
            if diff_text:
                print(diff_text)

        stats["chunk_details"].append(chunk_detail)

        if final_state["status"] == "success":
            new_code = final_state["modernized_code"].encode("utf-8")
            working_source = (
                working_source[:chunk.start_byte]
                + new_code
                + working_source[chunk.end_byte:]
            )
            for m in final_state.get("required_imports", []):
                if m not in collected_imports:
                    collected_imports.append(m)
            if final_state.get("risk_flag"):
                reason = final_state.get("risk_reason", "")
                print(f"    RISK FLAGGED: {reason}")
                stats["risk_flagged"].append({
                    "kind": chunk.kind, "start_byte": chunk.start_byte, "reason": reason,
                })
            if final_state.get("security_flag"):
                findings = final_state.get("security_findings", [])
                print(f"    SECURITY FLAGGED: {len(findings)} finding(s)")
                for f in findings:
                    print(f"      - {f['rule_id']} (line {f['line']}): {f['message']}")
                stats["security_flagged"].append({
                    "kind": chunk.kind, "start_byte": chunk.start_byte, "findings": findings,
                })
            stats["chunks_succeeded"] += 1
        else:
            print(f"SKIPPED (gave up): {chunk.kind} at byte {chunk.start_byte}")
            print(f"last error:\n{final_state['compiler_stderr']}")
            stats["chunks_gave_up"] += 1

    succeeded = stats["chunks_succeeded"]
    already_modern_count = stats["chunks_already_modern"]

    if succeeded == 0:
        if already_modern_count == len(chunks):
            print(f"\nAll {len(chunks)} chunk(s) were already modern — nothing to write.")
        else:
            print("\nNo chunks were successfully modernized. Exiting.")
        stats["duration_seconds"] = time.time() - start_time
        return stats

    existing_text = working_source.decode("utf-8")
    missing_imports = [m for m in collected_imports if not handler.has_import(existing_text, m)]
    if missing_imports:
        header_block = "".join(handler.import_statement(m) for m in missing_imports)
        working_source = header_block.encode("utf-8") + working_source

    new_source = working_source
    root, ext = os.path.splitext(file_path)
    output_path = f"{root}.modernized{ext}"
    with open(output_path, "wb") as f:
        f.write(new_source)
    stats["output_path"] = output_path
    print(
        f"\nWrote modernized file to {output_path} "
        f"({succeeded}/{len(chunks)} chunks modernized, "
        f"{already_modern_count} already modern/skipped)"
    )
    if stats["risk_flagged"]:
        print(f"\n{len(stats['risk_flagged'])} chunk(s) flagged for manual review:")
        for r in stats["risk_flagged"]:
            print(f"  - {r['kind']} at byte {r['start_byte']}: {r['reason']}")
    if stats["security_flagged"]:
        print(f"\n{len(stats['security_flagged'])} chunk(s) flagged by security scan:")
        for s in stats["security_flagged"]:
            for f in s["findings"]:
                print(f"  - {s['kind']} at byte {s['start_byte']}: {f['rule_id']} — {f['message']}")

    # Each chunk was only ever verified against its immediate predecessor
    # at the time it was modernized — never against the FINAL combination
    # of every successful chunk together. One last whole-file check closes
    # that gap before anything gets treated as trustworthy enough to PR.
    print("\n--- Final check: complete file vs. original behavior ---")
    original_result = verify(source.decode("utf-8"), handler.sandbox_filename, handler.run_command())
    final_result = verify(new_source.decode("utf-8"), handler.sandbox_filename, handler.run_command())

    final_check_passed = (
        original_result["status"] == "success"
        and final_result["status"] == "success"
        and original_result["stdout"] == final_result["stdout"]
    )
    stats["final_check_passed"] = final_check_passed
    if final_check_passed:
        print("PASSED — modernized file runs and matches original output.")
    else:
        print("FAILED — the combined file does not behave identically to the original.")
        print(f"original: {original_result}")
        print(f"final:    {final_result}")

    if open_pr and standalone_pr:
        if not final_check_passed:
            print("\nRefusing to open a PR: final behavioral check failed. "
                  "Inspect the output file manually before proposing it.")
            stats["duration_seconds"] = time.time() - start_time
            return stats

        eligible, reason = is_eligible(
            handler.name, _history, _MIN_TRACK_RECORD_CHUNKS, _MIN_SUCCESS_RATE
        )
        if not eligible:
            print(f"\nRefusing to open a PR: {reason}. "
                  "Run without --pr to build up track record first, or "
                  "lower the thresholds in .modernizer.toml's [autonomy] table.")
            stats["duration_seconds"] = time.time() - start_time
            return stats

        url = open_modernization_pr(
            file_path=file_path,
            new_content=new_source.decode("utf-8"),
            branch_name=f"chore/modernize-{file_path.replace('/', '-')}",
            pr_title=f"chore: modernize {file_path}",
            pr_body=(
                f"Automated modernization via code-modernizer.\n\n"
                f"{succeeded}/{len(chunks)} chunks successfully modernized "
                f"and verified in an isolated sandbox "
                f"({already_modern_count} already modern, skipped).\n\n"
                + _format_flags_section(stats["risk_flagged"], stats["security_flagged"])
            ),
        )
        stats["pr_url"] = url
        print(f"Opened PR: {url}")

    stats["duration_seconds"] = time.time() - start_time
    return stats


def _load_gitignore_spec(root_dir: str):
    """Load root_dir/.gitignore as a pathspec matcher, or None if there
    isn't one. Only the ROOT .gitignore is read — real git also honors
    nested per-directory .gitignore files, which this doesn't attempt;
    that covers the overwhelming majority of real projects (a single
    root-level .gitignore) without the complexity of replicating git's
    full nested-precedence rules."""
    gitignore_path = os.path.join(root_dir, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None
    import pathspec
    with open(gitignore_path) as f:
        return pathspec.PathSpec.from_lines("gitignore", f)


def discover_files(root_dir: str) -> list[str]:
    """Walk root_dir, returning every file with a registered language
    extension. Excludes whatever root_dir/.gitignore excludes, if one
    exists — falls back to a hardcoded vendor/build/VCS directory list
    otherwise. .git itself is always skipped regardless, since git
    doesn't apply .gitignore to itself but you never want to modernize
    anything in there."""
    from languages import get_handler

    spec = _load_gitignore_spec(root_dir)

    found = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        if spec is not None:
            dirnames[:] = [
                d for d in dirnames
                if not spec.match_file(os.path.relpath(os.path.join(dirpath, d), root_dir) + "/")
            ]
        else:
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]

        for filename in filenames:
            if ".modernized." in filename:
                # This tool's own output — same extension as its input,
                # so get_handler() would happily match it too. Without
                # this, re-running repo mode would treat every previous
                # run's output as new input and modernize it again.
                continue
            full_path = os.path.join(dirpath, filename)
            if spec is not None and spec.match_file(os.path.relpath(full_path, root_dir)):
                continue
            try:
                get_handler(full_path)
            except ValueError:
                continue
            found.append(full_path)
    return sorted(found)


def _read_all_file_contents(files: list[str]) -> dict[str, bytes]:
    """Pre-read every discovered file ONCE, so each file's probe
    generation can search every OTHER file for real call sites without
    re-reading repeatedly (O(n) reads total, not O(n^2))."""
    contents = {}
    for fp in files:
        try:
            with open(fp, "rb") as f:
                contents[fp] = f.read()
        except OSError as e:
            print(f"WARNING: could not read {fp} for cross-file probe context: {e}")
    return contents


def _run_files_concurrently(
    files: list[str], open_pr: bool, max_iterations: int, workers: int, file_contents: dict[str, bytes],
) -> list[dict]:
    """Run run_file() for independent files in parallel. Safe because
    files don't share state — only chunks WITHIN a single file have an
    ordering dependency (each verified against the accumulated result of
    prior chunks in that same file), and that accumulation is entirely
    local to one run_file() call. Results are reassembled in ORIGINAL
    file order (not completion order), so --report output stays
    deterministic across runs regardless of which file happened to
    finish first."""
    results: list[dict | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {}
        for i, file_path in enumerate(files):
            siblings = [content for fp, content in file_contents.items() if fp != file_path]
            future = executor.submit(run_file, file_path, open_pr, max_iterations, False, siblings)
            future_to_index[future] = i
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            file_path = files[i]
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"ERROR processing {file_path}: {e}")
                results[i] = {"file_path": file_path, "error": str(e)}
    return results


def run_repo(root_dir: str, open_pr: bool, max_iterations: int, workers: int = 1) -> list[dict]:
    files = discover_files(root_dir)
    print(f"Found {len(files)} modernizable file(s) under {root_dir}")

    file_contents = _read_all_file_contents(files)

    if workers > 1:
        print(
            f"Processing with {workers} concurrent workers. Files are "
            f"independent (only chunks WITHIN a file have ordering "
            f"dependencies via accumulated state), so this is safe — but "
            f"console output from different files WILL interleave. Use "
            f"--report for clean structured results instead of reading "
            f"the terminal for a concurrent run."
        )
        all_stats = _run_files_concurrently(files, open_pr, max_iterations, workers, file_contents)
    else:
        all_stats = []
        for i, file_path in enumerate(files, 1):
            print(f"\n{'=' * 60}\n[{i}/{len(files)}] {file_path}\n{'=' * 60}")
            siblings = [content for fp, content in file_contents.items() if fp != file_path]
            try:
                # standalone_pr=False: never open a per-file PR here, even
                # if open_pr is set — repo mode opens ONE combined PR below.
                stats = run_file(
                    file_path, open_pr, max_iterations,
                    standalone_pr=False, sibling_sources=siblings,
                )
            except Exception as e:
                print(f"ERROR processing {file_path}: {e}")
                stats = {"file_path": file_path, "error": str(e)}
            all_stats.append(stats)

    print(f"\n{'=' * 60}\nRepo summary ({len(files)} file(s))\n{'=' * 60}")
    total_succeeded = sum(s.get("chunks_succeeded", 0) for s in all_stats)
    total_chunks = sum(s.get("chunks_total", 0) for s in all_stats)
    total_risk = sum(len(s.get("risk_flagged", [])) for s in all_stats)
    total_security = sum(len(s.get("security_flagged", [])) for s in all_stats)
    files_written = sum(1 for s in all_stats if s.get("output_path"))
    print(f"Chunks modernized: {total_succeeded}/{total_chunks}")
    print(f"Files with output written: {files_written}/{len(files)}")
    print(f"Chunks flagged for manual review: {total_risk}")
    print(f"Chunks flagged by security scan: {total_security}")

    if open_pr:
        pr_url = _open_combined_pr(root_dir, all_stats)
        if pr_url:
            print(f"Opened combined PR: {pr_url}")
            for s in all_stats:
                if s.get("output_path"):
                    s["pr_url"] = pr_url

    return all_stats


def _open_combined_pr(root_dir: str, all_stats: list[dict]) -> str | None:
    """Gather every file whose final behavioral check passed AND whose
    language has a proven-enough track record, and open ONE PR containing
    all of them, instead of one PR per file. A repo can span languages
    with very different track records (e.g. Python well-proven, PHP
    brand new to this tool) — each file's eligibility is checked against
    ITS OWN language, not the repo as a whole."""
    checked_passed = [s for s in all_stats if s.get("final_check_passed") and s.get("output_path")]
    if not checked_passed:
        print("\nNo files passed their final check — no PR to open.")
        return None

    eligible = []
    for s in checked_passed:
        ok, reason = is_eligible(s["language"], _history, _MIN_TRACK_RECORD_CHUNKS, _MIN_SUCCESS_RATE)
        if ok:
            eligible.append(s)
        else:
            print(f"Excluding {s['file_path']} from PR: {reason}")

    if not eligible:
        print("\nNo files' languages have enough track record yet — no PR to open. "
              "Run without --pr to build up track record first.")
        return None

    files_for_pr = []
    for s in eligible:
        with open(s["output_path"], "rb") as f:
            content = f.read().decode("utf-8")
        files_for_pr.append((s["file_path"], content))

    all_risk_flagged = [r for s in eligible for r in s.get("risk_flagged", [])]
    all_security_flagged = [f for s in eligible for f in s.get("security_flagged", [])]
    skipped = len(all_stats) - len(eligible)
    branch_name = f"chore/modernize-repo-{int(time.time())}"

    return open_multi_file_pr(
        files=files_for_pr,
        branch_name=branch_name,
        pr_title=f"chore: modernize {len(eligible)} file(s)",
        pr_body=(
            f"Automated repo-wide modernization via code-modernizer.\n\n"
            f"{len(eligible)} file(s) included "
            f"({sum(s['chunks_succeeded'] for s in eligible)} chunks modernized total).\n"
            + (f"{skipped} file(s) skipped (failed their final check or language "
               f"track record).\n" if skipped else "")
            + "\n" + _format_flags_section(all_risk_flagged, all_security_flagged)
        ),
    )


def write_report(report_path: str, mode: str, results: list[dict]) -> None:
    """Dump per-chunk stats (iterations, status, timing, escalation/probe/
    risk usage) to JSON. Lets you actually see whether prompt tweaks are
    helping across runs instead of eyeballing one terminal output at a
    time, which is how every quality issue this session got found by
    hand — this turns that into something you can diff and aggregate."""
    import json
    from datetime import datetime, timezone

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote run report to {report_path}")


if __name__ == "__main__":
    # CLI flag defaults come from .modernizer.toml's [settings] table when
    # present, falling back to the hardcoded values below — an explicit
    # flag on the command line always overrides both.
    _settings = _config.get("settings", {})

    parser = argparse.ArgumentParser(description="Automated Code Modernization Engine")
    parser.add_argument("path", help="Path to a legacy source file, OR a directory to modernize recursively")
    parser.add_argument("--pr", action="store_true", help="Open a GitHub PR for each modernized file")
    parser.add_argument("--max-iterations", type=int, default=_settings.get("max_iterations", 5))
    parser.add_argument(
        "--report", metavar="PATH", default=_settings.get("report"),
        help="Write a structured JSON run report to this path",
    )
    parser.add_argument(
        "--workers", type=int, default=_settings.get("workers", 1),
        help="Process this many files concurrently in repo mode (default 1 = sequential). "
             "Each worker runs its own Docker containers and LLM calls — mind your machine's "
             "CPU/memory and Ollama's own concurrency limits before raising this.",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        results = run_repo(args.path, args.pr, args.max_iterations, args.workers)
        mode = "repo"
    else:
        results = [run_file(args.path, args.pr, args.max_iterations)]
        mode = "file"

    # Fold THIS run's per-language results into the persisted track
    # record for NEXT time. Deliberately happens after every run, whether
    # or not --pr was used — track record should reflect all attempts,
    # not just ones that already passed the gate, otherwise a language
    # could never accumulate the history needed to become eligible.
    updated_history = _history
    for s in results:
        language = s.get("language")
        if not language:
            continue
        attempted = s.get("chunks_succeeded", 0) + s.get("chunks_gave_up", 0)
        if attempted == 0:
            continue
        updated_history = record_run(updated_history, language, s.get("chunks_succeeded", 0), attempted)
    if updated_history != _history:
        save_history(updated_history)

    if args.report:
        write_report(args.report, mode, results)
