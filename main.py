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
from agents.review_graph import resume_review
from agents.nodes import llm_budget
from sandbox.verifier import verify
from git_ops.pr import open_modernization_pr, open_multi_file_pr
from track_record import load_history, save_history, record_run, is_eligible
from llm_budget import BudgetExceededError
from target_tests import detect_test_command, run_test_command, run_target_tests_on_copy
from regression_tests import generate_regression_test_file, regression_test_filename

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

# Set by run_repo() when --run-target-tests is used, read by write_report
# and the __main__ summary print. A module-level slot rather than a
# return value: run_repo's return type (list[dict], one per file) is
# already relied on by callers, and this is a single REPO-LEVEL result,
# not a per-file one — same reasoning as llm_budget being a
# module-level object rather than threaded through every call.
_last_target_test_result: dict | None = None


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


def _format_flags_section(
    risk_flagged: list[dict], security_flagged: list[dict], low_confidence_flagged: list[dict] | None = None,
) -> str:
    """Shared by the single-file PR body and (via the same shape) the
    combined repo-mode PR body — independent flag types, surfaced
    separately since they catch different things: risk_flag is "this
    touches something a stdout-diff can't fully verify" (LLM
    self-critique), security_flag is "static analysis found a specific
    pattern" (semgrep), low_confidence (mutation check) is "this chunk's
    OWN verification didn't distinguish a deliberately-broken variant
    from the accepted one" — a signal about the check's strength, not a
    property of the code itself. A chunk can be flagged by any
    combination of the three, or none."""
    low_confidence_flagged = low_confidence_flagged or []
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

    if low_confidence_flagged:
        parts.append(
            f"**{len(low_confidence_flagged)} chunk(s) flagged as low-confidence** "
            f"(a deliberately-broken mutant passed this chunk's own baseline/probe "
            f"checks — those checks may not have enough coverage to catch a subtle "
            f"regression here):\n"
            + "\n".join(f"- {c['kind']} at byte {c['start_byte']}" for c in low_confidence_flagged)
        )
    else:
        parts.append("No chunks flagged as low-confidence by the mutation check.")

    return "\n\n".join(parts)


def _isolate_probe_baselines(handler, chunks: list[dict]) -> list[dict]:
    """The `baseline_stdout` recorded on each probe during normal
    verification is NOT the probe's own isolated output — _verify_candidate
    runs the WHOLE candidate file with the probe appended (needed to prove
    the chunk doesn't break the rest of the file when spliced in), so that
    baseline is contaminated by the original file's own top-level side
    effects (confirmed by a real run: a file with its own `print(add(2,3))`
    at the bottom produced baseline_stdout='5\\n5\\n' for a probe that
    itself only prints once). A durable regression test should isolate the
    function under test, not replicate that whole-file artifact — so this
    re-runs each probe against JUST the modernized function (already
    proven correct) to capture a clean, function-only baseline instead of
    reusing the recorded one. One extra sandbox call per probe, only when
    --generate-regression-tests is on; skipped entirely otherwise."""
    isolated = []
    for chunk in chunks:
        clean_probes = []
        for probe in chunk["probes"]:
            candidate = (
                chunk["modernized_code"].encode("utf-8") + b"\n" + probe["snippet"].encode("utf-8") + b"\n"
            )
            result = verify(candidate.decode("utf-8"), handler.sandbox_filename, handler.run_command())
            if result["status"] == "success":
                clean_probes.append({"snippet": probe["snippet"], "baseline_stdout": result["stdout"]})
        isolated.append({"modernized_code": chunk["modernized_code"], "probes": clean_probes})
    return isolated


def _resolve_interactive_review(final_state: dict) -> dict:
    """Synchronously prompt the user right here in the terminal for a
    chunk modernize() paused on (status "awaiting_review"), then resume
    the SAME LangGraph review thread with their decision — all within
    this one CLI invocation, no separate command needed later. Returns
    final_state with status rewritten to "success" (approved) or
    "gave_up" (rejected), matching the two statuses every other code
    path already handles, so nothing downstream needs to know
    interactive mode was involved at all."""
    print("\n--- Chunk flagged for review ---")
    if final_state.get("risk_flag"):
        print(f"  RISK: {final_state.get('risk_reason', '')}")
    if final_state.get("security_flag"):
        for f in final_state.get("security_findings", []):
            print(f"  SECURITY: {f['rule_id']} (line {f['line']}): {f['message']}")
    if final_state.get("mutation_confidence_flag"):
        print(f"  LOW CONFIDENCE: {final_state.get('mutation_confidence_reason', '')}")
    print(f"\n{final_state['modernized_code']}\n")
    answer = input("Approve this chunk? [y/N]: ").strip().lower()
    approved = answer in ("y", "yes")
    resume_review(final_state["review_thread_id"], approved=approved)
    if approved:
        return {**final_state, "status": "success"}
    return {**final_state, "status": "gave_up", "compiler_stderr": "Rejected during interactive review."}


def run_file(
    file_path: str, open_pr: bool, max_iterations: int,
    standalone_pr: bool = True, sibling_sources: list[bytes] | None = None,
    generate_regression_tests: bool = False, interactive: bool = False,
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
    if not chunks and source.strip():
        # Zero chunks from a non-empty file is ambiguous: it's the
        # correct, silent outcome for a file with no functions/methods at
        # all (e.g. pure constants/config), but it's ALSO what a
        # tree-sitter parse failure on malformed/binary-ish input looks
        # like — same return value, no exception raised either way. A
        # user watching the run has no way to tell those apart without
        # this nudge to go check manually.
        print(
            f"    (warning: no functions/methods found — if this file isn't "
            f"actually empty of them, tree-sitter may have failed to parse it)"
        )

    stats = {
        "file_path": file_path,
        "language": handler.name,
        "chunks_total": len(chunks),
        "chunks_succeeded": 0,
        "chunks_already_modern": 0,
        "chunks_gave_up": 0,
        "chunks_budget_exceeded": 0,
        "risk_flagged": [],
        "security_flagged": [],
        "low_confidence_flagged": [],
        "chunk_details": [],
        "output_path": None,
        "final_check_passed": None,
        "pr_url": None,
        "regression_test_path": None,
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
    successful_chunks_for_regression_tests: list[dict] = []
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

        try:
            final_state = modernize(
                handler.name, working_source, chunk.start_byte, chunk.end_byte,
                max_iterations=max_iterations, sibling_sources=sibling_sources,
                interactive=interactive,
            )
            if final_state["status"] == "awaiting_review":
                final_state = _resolve_interactive_review(final_state)
        except BudgetExceededError as e:
            # Stop processing THIS file's remaining chunks entirely rather
            # than let the exception propagate and crash the run (single-
            # file mode has no caller to catch it) or, worse, silently
            # keep looping and hitting the identical error on every
            # remaining chunk. The chunks already modernized before this
            # point are kept — only what's left unprocessed is skipped.
            print(f"STOPPED: {e}")
            remaining = [c for c in chunks if c.start_byte <= chunk.start_byte]
            stats["chunks_budget_exceeded"] += len(remaining)
            for c in remaining:
                stats["chunk_details"].append({
                    "kind": c.kind, "start_byte": c.start_byte,
                    "status": "budget_exceeded", "iterations": 0,
                    "duration_seconds": 0.0,
                })
            break
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
            "mutation_confidence_flag": final_state.get("mutation_confidence_flag", False),
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
            successful_chunks_for_regression_tests.append({
                "modernized_code": final_state["modernized_code"],
                "probes": final_state.get("probes") or [],
            })
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
            if final_state.get("mutation_confidence_flag"):
                print(f"    LOW CONFIDENCE: {final_state.get('mutation_confidence_reason', '')}")
                stats["low_confidence_flagged"].append({
                    "kind": chunk.kind, "start_byte": chunk.start_byte,
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

    if generate_regression_tests:
        isolated_chunks = _isolate_probe_baselines(handler, successful_chunks_for_regression_tests)
        test_source = generate_regression_test_file(handler.name, isolated_chunks)
        if test_source is None:
            print("\n--generate-regression-tests: no probes captured for this file — nothing to write.")
        else:
            test_path = regression_test_filename(handler.name, file_path)
            with open(test_path, "w") as f:
                f.write(test_source)
            stats["regression_test_path"] = test_path
            print(f"Wrote durable regression test to {test_path} (from captured verification probes)")

    if stats["risk_flagged"]:
        print(f"\n{len(stats['risk_flagged'])} chunk(s) flagged for manual review:")
        for r in stats["risk_flagged"]:
            print(f"  - {r['kind']} at byte {r['start_byte']}: {r['reason']}")
    if stats["security_flagged"]:
        print(f"\n{len(stats['security_flagged'])} chunk(s) flagged by security scan:")
        for s in stats["security_flagged"]:
            for f in s["findings"]:
                print(f"  - {s['kind']} at byte {s['start_byte']}: {f['rule_id']} — {f['message']}")
    if stats["low_confidence_flagged"]:
        print(f"\n{len(stats['low_confidence_flagged'])} chunk(s) flagged as low-confidence (mutation check):")
        for c in stats["low_confidence_flagged"]:
            print(f"  - {c['kind']} at byte {c['start_byte']}")

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
                + _format_flags_section(stats["risk_flagged"], stats["security_flagged"], stats["low_confidence_flagged"])
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
    generate_regression_tests: bool = False,
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
            future = executor.submit(
                run_file, file_path, open_pr, max_iterations, False, siblings, generate_regression_tests,
            )
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


def run_repo(
    root_dir: str, open_pr: bool, max_iterations: int, workers: int = 1, run_target_tests: bool = False,
    generate_regression_tests: bool = False, interactive: bool = False,
) -> list[dict]:
    global _last_target_test_result
    _last_target_test_result = None
    if interactive and workers > 1:
        # A synchronous input() prompt from N concurrent threads at once
        # is a genuinely broken UX (garbled, unpredictable terminal
        # output, no way to tell which file a prompt belongs to) — not
        # just noisier than the existing "console output WILL interleave"
        # warning for concurrent mode, actually unusable. Refuse clearly
        # rather than let it silently misbehave.
        raise ValueError("--interactive is not supported with --workers > 1 (concurrent prompts would garble the terminal). Run with --workers 1.")
    files = discover_files(root_dir)
    print(f"Found {len(files)} modernizable file(s) under {root_dir}")

    file_contents = _read_all_file_contents(files)

    baseline_test_result = None
    test_framework = None
    if run_target_tests:
        detected = detect_test_command(root_dir)
        if detected is None:
            print("\n--run-target-tests: no recognized test framework found at repo root — skipping.")
        else:
            test_framework, test_command = detected
            print(f"\n--run-target-tests: running {test_framework}'s existing test suite as a baseline "
                  f"(BEFORE modernization, on the untouched repo)...")
            baseline_test_result = run_test_command(root_dir, test_command)
            print(f"Baseline: {baseline_test_result['status']}")

    if workers > 1:
        print(
            f"Processing with {workers} concurrent workers. Files are "
            f"independent (only chunks WITHIN a file have ordering "
            f"dependencies via accumulated state), so this is safe — but "
            f"console output from different files WILL interleave. Use "
            f"--report for clean structured results instead of reading "
            f"the terminal for a concurrent run."
        )
        all_stats = _run_files_concurrently(
            files, open_pr, max_iterations, workers, file_contents, generate_regression_tests,
        )
    else:
        all_stats = []
        for i, file_path in enumerate(files, 1):
            if llm_budget.is_exceeded():
                # Stop BEFORE starting a new file, not after hitting the
                # same BudgetExceededError on its very first chunk — the
                # remaining files are cleanly reported as never attempted
                # rather than each showing up as its own "ERROR".
                remaining = len(files) - i + 1
                print(f"\nLLM call budget exhausted — stopping before {remaining} remaining file(s).")
                for skipped_path in files[i - 1:]:
                    all_stats.append({"file_path": skipped_path, "error": "budget_exceeded"})
                break
            print(f"\n{'=' * 60}\n[{i}/{len(files)}] {file_path}\n{'=' * 60}")
            siblings = [content for fp, content in file_contents.items() if fp != file_path]
            try:
                # standalone_pr=False: never open a per-file PR here, even
                # if open_pr is set — repo mode opens ONE combined PR below.
                stats = run_file(
                    file_path, open_pr, max_iterations,
                    standalone_pr=False, sibling_sources=siblings,
                    generate_regression_tests=generate_regression_tests,
                    interactive=interactive,
                )
            except Exception as e:
                print(f"ERROR processing {file_path}: {e}")
                stats = {"file_path": file_path, "error": str(e)}
            all_stats.append(stats)

    if run_target_tests and test_framework is not None:
        overlay_files = {}
        for s in all_stats:
            if not s.get("output_path"):
                continue
            try:
                with open(s["output_path"]) as f:
                    overlay_files[os.path.relpath(s["file_path"], root_dir)] = f.read()
            except OSError as e:
                # A file written earlier in this run being gone/unreadable
                # by the time we get here (moved, deleted, permissions
                # changed mid-run) shouldn't crash the whole repo summary
                # — degrade to "not included in this test comparison"
                # instead, same fail-soft spirit as run_test_command's
                # own error handling.
                print(f"    (warning: could not read {s['output_path']} for target-test overlay: {e})")
        if not overlay_files:
            print(f"\n--run-target-tests: no files were successfully modernized — nothing to re-test.")
        else:
            print(
                f"\n--run-target-tests: re-running {test_framework}'s test suite against a TEMPORARY "
                f"copy of the repo with {len(overlay_files)} modernized file(s) overlaid "
                f"(your real working tree is never touched)..."
            )
            after_result = run_target_tests_on_copy(root_dir, overlay_files)
            regressed = (
                baseline_test_result is not None and after_result is not None
                and baseline_test_result["status"] == "success" and after_result["status"] != "success"
            )
            print(f"After: {after_result['status'] if after_result else 'error'}"
                  + (" — REGRESSION vs baseline" if regressed else ""))
            _last_target_test_result = {
                "framework": test_framework,
                "baseline": baseline_test_result,
                "after": after_result,
                "regressed": regressed,
            }

    print(f"\n{'=' * 60}\nRepo summary ({len(files)} file(s))\n{'=' * 60}")
    total_succeeded = sum(s.get("chunks_succeeded", 0) for s in all_stats)
    total_chunks = sum(s.get("chunks_total", 0) for s in all_stats)
    total_risk = sum(len(s.get("risk_flagged", [])) for s in all_stats)
    total_security = sum(len(s.get("security_flagged", [])) for s in all_stats)
    total_low_confidence = sum(len(s.get("low_confidence_flagged", [])) for s in all_stats)
    files_written = sum(1 for s in all_stats if s.get("output_path"))
    print(f"Chunks modernized: {total_succeeded}/{total_chunks}")
    print(f"Files with output written: {files_written}/{len(files)}")
    print(f"Chunks flagged for manual review: {total_risk}")
    print(f"Chunks flagged by security scan: {total_security}")
    print(f"Chunks flagged as low-confidence (mutation check): {total_low_confidence}")

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
    all of them, instead of one PR per file.

    A repo can span languages with very different track records (e.g.
    Python well-proven, PHP brand new to this tool) — each file's
    eligibility is checked against ITS OWN language, not the repo as a
    whole.

    If --run-target-tests found a regression (the target repo's OWN
    pre-existing test suite passed before modernization and doesn't
    after), refuse the PR outright — an independent, human-authored
    oracle is a strictly stronger signal than anything this project
    generates itself, so it gates PR eligibility the same way
    final_check_passed does."""
    if _last_target_test_result and _last_target_test_result.get("regressed"):
        print(
            "\nRefusing to open a PR: --run-target-tests found a regression "
            f"({_last_target_test_result['framework']}'s test suite passed before "
            "modernization and doesn't after). Inspect the *.modernized.* files "
            "manually before proposing them."
        )
        return None

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
    all_low_confidence_flagged = [c for s in eligible for c in s.get("low_confidence_flagged", [])]
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
            + "\n" + _format_flags_section(all_risk_flagged, all_security_flagged, all_low_confidence_flagged)
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
        "llm_usage": llm_budget.summary(),
        "target_test_result": _last_target_test_result,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote run report to {report_path}")


def _html_escape(text) -> str:
    import html
    return html.escape(str(text), quote=True)


_STATUS_BADGE_CLASS = {
    "success": "badge-success",
    "already_modern": "badge-neutral",
    "gave_up": "badge-fail",
    "budget_exceeded": "badge-warn",
    "failed": "badge-fail",
}


def write_html_report(report_path: str, mode: str, results: list[dict]) -> None:
    """Human-readable counterpart to write_report's raw JSON — a
    self-contained (no external assets, works offline) HTML page a
    non-CLI reviewer can open in a browser to see, per chunk: pass/fail,
    every flag this project's verification pipeline raised (risk,
    security, low-confidence-mutation), iteration/escalation cost, and
    the actual diff. The JSON report already carries all of this data;
    this is pure presentation on top of it — no pipeline changes, so it
    carries none of the pipeline's own risk. Written whenever
    --report-html is passed, independent of (and in addition to)
    --report — the two serve different audiences (tooling vs. a human
    skimming what an autonomous run actually did before trusting a PR
    made from it)."""
    from datetime import datetime, timezone

    generated_at = datetime.now(timezone.utc).isoformat()
    usage = llm_budget.summary()

    total_chunks = sum(s.get("chunks_total", 0) for s in results if "error" not in s)
    total_succeeded = sum(s.get("chunks_succeeded", 0) for s in results if "error" not in s)
    total_flagged = sum(
        len(s.get("risk_flagged", [])) + len(s.get("security_flagged", [])) + len(s.get("low_confidence_flagged", []))
        for s in results if "error" not in s
    )

    file_sections = []
    for s in results:
        if "error" in s:
            file_sections.append(
                f'<section class="file-card"><h2>{_html_escape(s.get("file_path", "?"))}</h2>'
                f'<p class="badge badge-fail">ERROR</p><pre class="err">{_html_escape(s["error"])}</pre></section>'
            )
            continue

        rows = []
        for c in s.get("chunk_details", []):
            badge_class = _STATUS_BADGE_CLASS.get(c.get("status"), "badge-neutral")
            flags = []
            if c.get("risk_flag"):
                flags.append('<span class="badge badge-warn">risk</span>')
            if c.get("security_flag"):
                flags.append('<span class="badge badge-warn">security</span>')
            if c.get("mutation_confidence_flag"):
                flags.append('<span class="badge badge-warn">low-confidence</span>')
            if c.get("used_escalation"):
                flags.append('<span class="badge badge-neutral">escalated</span>')
            diff_html = (
                f'<details><summary>diff</summary><pre class="diff">{_html_escape(c["diff"])}</pre></details>'
                if c.get("diff") else ""
            )
            rows.append(
                "<tr>"
                f'<td>{_html_escape(c.get("kind", ""))} @{c.get("start_byte", "?")}</td>'
                f'<td><span class="badge {badge_class}">{_html_escape(c.get("status", ""))}</span></td>'
                f'<td>{c.get("iterations", 0)}</td>'
                f'<td>{" ".join(flags)}</td>'
                f'<td>{c.get("duration_seconds", 0):.1f}s</td>'
                f'<td>{diff_html}</td>'
                "</tr>"
            )

        file_sections.append(
            f'<section class="file-card"><h2>{_html_escape(s.get("file_path", "?"))} '
            f'<span class="lang">{_html_escape(s.get("language", ""))}</span></h2>'
            f'<p>{s.get("chunks_succeeded", 0)}/{s.get("chunks_total", 0)} chunk(s) succeeded, '
            f'{s.get("chunks_already_modern", 0)} already modern, '
            f'{s.get("chunks_gave_up", 0)} gave up</p>'
            f'<table><thead><tr><th>Chunk</th><th>Status</th><th>Iters</th>'
            f'<th>Flags</th><th>Time</th><th>Diff</th></tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>code-modernizer run report</title>
<style>
body {{ font: 14px/1.5 -apple-system, sans-serif; max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background: #fff; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; }}
.summary {{ display: flex; gap: 2rem; margin: 1rem 0 2rem; }}
.summary div {{ background: #f5f5f5; border-radius: 8px; padding: 0.75rem 1rem; }}
.summary .n {{ font-size: 1.5rem; font-weight: 600; display: block; }}
.file-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin-bottom: 1.5rem; }}
.lang {{ font-weight: 400; color: #888; font-size: 0.85rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th, td {{ text-align: left; padding: 0.4rem 0.5rem; border-bottom: 1px solid #eee; vertical-align: top; }}
.badge {{ display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px; font-size: 0.75rem; margin-right: 0.25rem; }}
.badge-success {{ background: #d4f4dd; color: #1a7a3a; }}
.badge-fail {{ background: #fbdada; color: #a3282f; }}
.badge-warn {{ background: #fff2cc; color: #97740e; }}
.badge-neutral {{ background: #e6e6e6; color: #555; }}
.diff, .err {{ white-space: pre-wrap; background: #f8f8f8; padding: 0.5rem; border-radius: 6px; overflow-x: auto; }}
</style></head><body>
<h1>code-modernizer run report</h1>
<p>Generated {_html_escape(generated_at)} &middot; mode: {_html_escape(mode)}</p>
<div class="summary">
<div><span class="n">{total_succeeded}/{total_chunks}</span>chunks succeeded</div>
<div><span class="n">{total_flagged}</span>flagged for review</div>
<div><span class="n">{usage['total_calls']}</span>LLM calls</div>
</div>
{''.join(file_sections)}
</body></html>"""

    with open(report_path, "w") as f:
        f.write(html_doc)
    print(f"Wrote human-readable HTML report to {report_path}")


def main():
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
        "--report-html", metavar="PATH", default=_settings.get("report_html"),
        help="Write a self-contained, human-readable HTML run report to this path "
             "(per-chunk status, flags, diffs) — for reviewing a run in a browser "
             "instead of reading raw JSON. Independent of --report; pass both to get both.",
    )
    parser.add_argument(
        "--workers", type=int, default=_settings.get("workers", 1),
        help="Process this many files concurrently in repo mode (default 1 = sequential). "
             "Each worker runs its own Docker containers and LLM calls — mind your machine's "
             "CPU/memory and Ollama's own concurrency limits before raising this.",
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=_settings.get("max_llm_calls_per_run"),
        help="Hard ceiling on total LLM calls for this run (default: unlimited). A circuit "
             "breaker for repo-mode runs, where best-of-N x retry-iterations x probes x "
             "chunks x files can multiply fast with nothing capping it otherwise — once hit, "
             "remaining chunks/files are cleanly skipped rather than the run crashing or "
             "silently continuing forever.",
    )
    parser.add_argument(
        "--run-target-tests", action="store_true", default=_settings.get("run_target_tests", False),
        help="Repo mode only. If the target repo has a recognized test suite (pytest, "
             "npm test, PHPUnit, Maven/Gradle), run it BEFORE modernization (baseline, "
             "untouched repo) and AFTER (a TEMPORARY COPY with modernized files overlaid — "
             "your real working tree is never touched). Unlike everything else in this "
             "project, this runs the test command directly on the HOST, not the sandbox: "
             "it needs the target repo's own installed dependencies, which only exist there. "
             "A regression (baseline passed, after doesn't) refuses --pr for this run.",
    )
    parser.add_argument(
        "--generate-regression-tests", action="store_true",
        default=_settings.get("generate_regression_tests", False),
        help="For each successfully modernized chunk, write its captured verification "
             "probes (real call sites + LLM-synthesized examples, already proven to match "
             "the original's behavior) into a standalone, durable test file next to the "
             "output — turning this run's verification into lasting test coverage that "
             "needs neither this project nor an LLM to re-check later. New sibling file "
             "only (e.g. test_calc_modernized.py) — never overwrites anything. Supported "
             "for python/javascript/typescript/php (the languages that generate probes at "
             "all); a no-op for cpp/java.",
    )
    parser.add_argument(
        "--interactive", action="store_true", default=_settings.get("interactive", False),
        help="Pause on any chunk flagged (risk, security, or low-confidence mutation check) "
             "and prompt right here in the terminal to approve or reject it, instead of just "
             "flagging and continuing. Uses LangGraph's native interrupt()/Command mechanism "
             "under the hood — approve keeps the chunk as a success, reject treats it like "
             "any other failed chunk (not written to output). Not supported with --workers > 1 "
             "(concurrent prompts would garble the terminal).",
    )
    args = parser.parse_args()

    if args.interactive and args.workers > 1:
        parser.error("--interactive is not supported with --workers > 1 (concurrent prompts would garble the terminal)")

    # Reset ONCE here, at the true top-level entry point — never inside
    # run_file/run_repo themselves, or a repo-wide budget would silently
    # reset per file instead of accumulating across the whole run (see
    # LLMBudget.reset's docstring).
    llm_budget.reset(max_calls=args.max_llm_calls)

    if os.path.isdir(args.path):
        results = run_repo(
            args.path, args.pr, args.max_iterations, args.workers,
            args.run_target_tests, args.generate_regression_tests, args.interactive,
        )
        mode = "repo"
    else:
        results = [run_file(
            args.path, args.pr, args.max_iterations,
            generate_regression_tests=args.generate_regression_tests,
            interactive=args.interactive,
        )]
        mode = "file"

    _usage = llm_budget.summary()
    print(
        f"\nLLM calls this run: {_usage['total_calls']}"
        + (f" (budget: {_usage['max_calls']})" if _usage["max_calls"] is not None else "")
    )
    if _usage["total_input_tokens"] or _usage["total_output_tokens"]:
        print(f"Tokens: {_usage['total_input_tokens']} in / {_usage['total_output_tokens']} out")
    if _usage["calls_by_model"]:
        print("By model: " + ", ".join(f"{m}={c}" for m, c in _usage["calls_by_model"].items()))

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
    if args.report_html:
        write_html_report(args.report_html, mode, results)


if __name__ == "__main__":
    main()
