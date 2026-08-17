import argparse
import dataclasses
import difflib
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import config
from config import load_config, apply_config_to_environment, load_recipes, load_profiles

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
from characterization_tests import generate_characterization_test_file, characterization_test_filename
from sarif_report import write_sarif_report

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


@dataclasses.dataclass(frozen=True)
class RunOptions:
    """Every run-wide behavior toggle, bundled into ONE object instead of
    threaded as individual parameters through run_file/run_repo/
    _run_files_concurrently/_run_file_in_worktree/watch_run. Before this,
    adding one new flag meant editing all 5 of those signatures AND every
    call site between them in lockstep — exactly the kind of change
    that's easy to get subtly wrong (confirmed: an earlier version of
    --characterize had a real early-return bug from exactly this kind of
    threading). Adding a new option now means adding one field here and
    reading it at the ONE place that needs it — every function that
    doesn't care about a given option doesn't need to know it exists.

    Frozen (immutable) deliberately: nothing should mutate a shared
    RunOptions instance after construction — main() builds exactly one
    from parsed CLI args and passes it down; mutating it partway through
    a run would be exactly the kind of implicit, hard-to-trace state
    change this refactor exists to avoid.

    Deliberately NOT included: file_path/root_dir, sibling_sources,
    standalone_pr. Those aren't run-wide BEHAVIOR — they're either
    per-call DATA (sibling_sources genuinely differs file-to-file within
    one run) or internal wiring (standalone_pr is always False except
    the single CLI single-file dispatch), not something a user sets
    once for a whole run the way every field below is."""
    open_pr: bool = False
    max_iterations: int = 5
    workers: int = 1
    run_target_tests: bool = False
    generate_regression_tests: bool = False
    interactive: bool = False
    recipe_instruction: str | None = None
    punt_check_enabled: bool = False
    characterize: bool = False
    isolate_workers: bool = False
    staged_only: bool = False

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


def _isolate_probe_baselines(handler, chunks: list[dict], source_field: str = "modernized_code") -> list[dict]:
    """The `baseline_stdout` recorded on each probe during normal
    verification is NOT the probe's own isolated output — _verify_candidate
    runs the WHOLE candidate file with the probe appended (needed to prove
    the chunk doesn't break the rest of the file when spliced in), so that
    baseline is contaminated by the original file's own top-level side
    effects (confirmed by a real run: a file with its own `print(add(2,3))`
    at the bottom produced baseline_stdout='5\\n5\\n' for a probe that
    itself only prints once). A durable regression/characterization test
    should isolate the function under test, not replicate that whole-file
    artifact — so this re-runs each probe against JUST the function
    (already proven correct against ITSELF, whichever version source_field
    names) to capture a clean, function-only baseline instead of reusing
    the recorded one. One extra sandbox call per probe, only when
    --generate-regression-tests or --characterize is on; skipped entirely
    otherwise.

    source_field selects which field of each chunk dict to isolate against:
    "modernized_code" (default, for --generate-regression-tests — the
    function ALREADY passed full verification against this exact source)
    or "original_code" (for --characterize — pinning behavior BEFORE any
    rewrite, so there's no "modernized" version to use yet)."""
    isolated = []
    for chunk in chunks:
        clean_probes = []
        for probe in chunk["probes"]:
            candidate = (
                chunk[source_field].encode("utf-8") + b"\n" + probe["snippet"].encode("utf-8") + b"\n"
            )
            result = verify(candidate.decode("utf-8"), handler.sandbox_filename, handler.run_command())
            if result["status"] == "success":
                clean_probes.append({"snippet": probe["snippet"], "baseline_stdout": result["stdout"]})
        isolated.append({source_field: chunk[source_field], "probes": clean_probes})
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


def _estimate_llm_calls_per_chunk(max_iterations: int) -> tuple[int, int]:
    """Rough (min, max) LLM-call range for ONE chunk that ends up
    actually needing modernization (already-modern chunks cost zero —
    see handler.already_modern's fast path). Deliberately a coarse
    estimate, not a simulation: --plan exists to answer "is this worth
    running" cheaply (no LLM/Docker calls at all), not to predict the
    exact bill.

    min: succeeds on the FIRST attempt — costs BEST_OF_N_ON_FIRST_ATTEMPT
    refactor calls (best-of-N always fires on that first attempt; see
    agents/nodes.py:refactorer_node) + 1 risk-assessment call + 1
    mutation-confidence call (both run once, post-success only — see
    agents/graph.py:modernize). No probe-generation call counted in the
    minimum: it's skipped whenever real call sites already cover
    MAX_PROBES_PER_CHUNK, which the static estimate can't know in
    advance either way, so this errs toward the cheaper case.

    max: exhausts every retry (max_iterations - 1 additional single-
    candidate attempts, since best-of-N only applies to the first) before
    finally succeeding, PLUS one probe-generation call."""
    from agents.nodes import BEST_OF_N_ON_FIRST_ATTEMPT
    min_calls = BEST_OF_N_ON_FIRST_ATTEMPT + 2
    max_calls = BEST_OF_N_ON_FIRST_ATTEMPT + max(0, max_iterations - 1) + 3
    return min_calls, max_calls


def plan_file(file_path: str, max_iterations: int) -> dict:
    """Chunk `file_path` and report what a real run would face, WITHOUT
    making any LLM or Docker call — pure tree-sitter parsing plus the
    already-modern regex/heuristic check (see languages/base.py), both
    fully local and instant. This is the entire point of --plan: let
    someone see the scope and rough cost of a run before spending
    anything on it."""
    handler = get_handler(file_path)
    with open(file_path, "rb") as f:
        source = f.read()
    chunks = handler.chunk(source)
    already_modern = sum(1 for c in chunks if handler.already_modern(c.code))
    to_modernize = len(chunks) - already_modern
    min_per, max_per = _estimate_llm_calls_per_chunk(max_iterations)
    return {
        "file_path": file_path,
        "language": handler.name,
        "chunks_total": len(chunks),
        "chunks_already_modern": already_modern,
        "chunks_to_modernize": to_modernize,
        "estimated_llm_calls_min": to_modernize * min_per,
        "estimated_llm_calls_max": to_modernize * max_per,
    }


def plan_run(path: str, max_iterations: int) -> list[dict]:
    """--plan entry point for both single-file and repo mode. Repo mode
    reuses discover_files so the estimate matches exactly what a real
    `run_repo` call would find (same .gitignore/extension filtering)."""
    files = discover_files(path) if os.path.isdir(path) else [path]
    return [plan_file(f, max_iterations) for f in files]


def print_plan(plans: list[dict]) -> None:
    total_chunks = sum(p["chunks_total"] for p in plans)
    total_already_modern = sum(p["chunks_already_modern"] for p in plans)
    total_to_modernize = sum(p["chunks_to_modernize"] for p in plans)
    total_min = sum(p["estimated_llm_calls_min"] for p in plans)
    total_max = sum(p["estimated_llm_calls_max"] for p in plans)

    print(f"\n--plan: {len(plans)} file(s) scanned, no LLM/Docker calls made\n")
    for p in plans:
        print(
            f"  [{p['language']}] {p['file_path']}: {p['chunks_total']} chunk(s) "
            f"({p['chunks_already_modern']} already modern, {p['chunks_to_modernize']} to modernize) "
            f"— est. {p['estimated_llm_calls_min']}-{p['estimated_llm_calls_max']} LLM calls"
        )
    print(
        f"\nTotal: {total_chunks} chunk(s), {total_already_modern} already modern, "
        f"{total_to_modernize} to modernize"
    )
    print(f"Estimated LLM calls for a real run: {total_min}-{total_max}")
    print(
        "(Range reflects best case: every chunk passes on its first attempt, vs. "
        "worst case: every chunk exhausts --max-iterations before succeeding. "
        "Actual sandbox/verification cost — Docker time, not LLM calls — isn't estimated here.)"
    )


def run_file(
    file_path: str, options: RunOptions,
    standalone_pr: bool = True, sibling_sources: list[bytes] | None = None,
) -> dict:
    """Modernize one file. Returns a stats dict (also the data source for
    --report) rather than just printing, so repo mode can aggregate
    results across many files without re-parsing terminal output.

    standalone_pr controls whether THIS call opens its own single-file
    PR when options.open_pr is set. Repo mode passes False here and
    instead collects every file's output itself, opening ONE combined
    multi-file PR at the end — GitHub has no atomic multi-file update via
    the single-file convenience API, so letting each file open its own PR
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
        "chunks_punted": 0,
        "chunks_budget_exceeded": 0,
        "risk_flagged": [],
        "security_flagged": [],
        "low_confidence_flagged": [],
        "chunk_details": [],
        "output_path": None,
        "final_check_passed": None,
        "pr_url": None,
        "regression_test_path": None,
        "characterization_test_path": None,
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
    all_chunks_for_characterization: list[dict] = []
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
                max_iterations=options.max_iterations, sibling_sources=sibling_sources,
                interactive=options.interactive, recipe_instruction=options.recipe_instruction,
                punt_check_enabled=options.punt_check_enabled,
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
            "used_deterministic_rule": final_state.get("used_deterministic_rule", False),
            "punted": final_state.get("punted", False),
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

        if options.characterize and final_state.get("probes"):
            # Captured regardless of final status (success OR gave_up) —
            # probes are captured against the ORIGINAL code before
            # refactorer_node ever runs (see agents/graph.py:modernize),
            # so a chunk this project couldn't successfully modernize
            # still leaves behind a durable record of what it used to do.
            # Excludes punted chunks (--punt-check skips probe capture
            # entirely — see agents/graph.py:_punted_initial_state) and
            # already-modern chunks (`continue`d above, never reach here;
            # nothing "legacy" to characterize about already-modern code).
            all_chunks_for_characterization.append({
                "original_code": chunk.code,
                "probes": final_state["probes"],
            })

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
                    "kind": chunk.kind, "start_byte": chunk.start_byte,
                    # 1-indexed line this chunk STARTS at in the real file —
                    # semgrep's own `line` on each finding is relative to the
                    # isolated single-chunk file it actually scanned (see
                    # scan_security's docstring), not the real file, so a
                    # SARIF report (sarif_report.py) needs this offset to
                    # report a real, useful line number.
                    "start_line": source[:chunk.start_byte].count(b"\n") + 1,
                    "findings": findings,
                })
            if final_state.get("mutation_confidence_flag"):
                print(f"    LOW CONFIDENCE: {final_state.get('mutation_confidence_reason', '')}")
                stats["low_confidence_flagged"].append({
                    "kind": chunk.kind, "start_byte": chunk.start_byte,
                })
            stats["chunks_succeeded"] += 1
        elif final_state.get("punted"):
            # Punted BEFORE any attempt (see --punt-check) — deliberately
            # NOT folded into chunks_gave_up: that counter feeds
            # track_record.py's success-rate calculation, and a chunk
            # that was never actually tried shouldn't count as a failed
            # attempt any more than an already-modern chunk does (see
            # record_run's docstring on why chunks_already_modern is
            # excluded the same way).
            print(f"SKIPPED (punted before attempting): {chunk.kind} at byte {chunk.start_byte}")
            print(f"reason:\n{final_state['compiler_stderr']}")
            stats["chunks_punted"] += 1
        else:
            print(f"SKIPPED (gave up): {chunk.kind} at byte {chunk.start_byte}")
            print(f"last error:\n{final_state['compiler_stderr']}")
            stats["chunks_gave_up"] += 1

    if options.characterize:
        # Deliberately BEFORE the succeeded==0 early return below —
        # characterization exists precisely FOR the case a file's
        # modernization completely failed: a permanent safety net for
        # the original behavior even when this project couldn't handle
        # any of it, not just a bonus for files that mostly succeeded.
        isolated_characterization_chunks = _isolate_probe_baselines(
            handler, all_chunks_for_characterization, source_field="original_code",
        )
        characterization_source = generate_characterization_test_file(handler.name, isolated_characterization_chunks)
        if characterization_source is None:
            print("\n--characterize: no probes captured for this file — nothing to write.")
        else:
            characterization_path = characterization_test_filename(handler.name, file_path)
            with open(characterization_path, "w") as f:
                f.write(characterization_source)
            stats["characterization_test_path"] = characterization_path
            print(f"Wrote characterization test to {characterization_path} (pins ORIGINAL behavior, pre-modernization)")

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

    if options.generate_regression_tests:
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

    if options.open_pr and standalone_pr:
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


def _run_file_in_worktree(
    root_dir: str, file_path: str, options: RunOptions, sibling_sources: list[bytes],
) -> dict:
    """Same job as run_file(), but the actual read/chunk/verify/write
    cycle runs against a dedicated git worktree checkout of root_dir
    (see git_ops/worktree.py's module docstring for exactly what this
    does and doesn't protect against) instead of the shared root_dir
    directly. Any output file(s) run_file() produced inside the worktree
    are copied back into root_dir before the worktree is torn down —
    root_dir itself is never written to until that explicit copy-back,
    and the worktree is always removed (success or failure) so a run
    doesn't leak temp checkouts."""
    from git_ops.worktree import create_worktree, remove_worktree

    rel_path = os.path.relpath(file_path, root_dir)
    worktree_path = create_worktree(root_dir)
    try:
        stats = run_file(
            os.path.join(worktree_path, rel_path),
            dataclasses.replace(options, interactive=False),
            standalone_pr=False, sibling_sources=sibling_sources,
        )
        # Report paths relative to the REAL tree, not the throwaway
        # worktree — everything downstream (--report, the combined PR,
        # --run-target-tests overlay) expects paths under root_dir.
        stats["file_path"] = file_path
        for path_key in ("output_path", "regression_test_path", "characterization_test_path"):
            wt_path = stats.get(path_key)
            if wt_path and os.path.isfile(wt_path):
                real_path = os.path.join(root_dir, os.path.relpath(wt_path, worktree_path))
                shutil.copy2(wt_path, real_path)
                stats[path_key] = real_path
        return stats
    finally:
        remove_worktree(root_dir, worktree_path)


def _run_files_concurrently(
    files: list[str], options: RunOptions, file_contents: dict[str, bytes],
    root_dir: str | None = None,
) -> list[dict]:
    """Run run_file() for independent files in parallel. Safe because
    files don't share state — only chunks WITHIN a single file have an
    ordering dependency (each verified against the accumulated result of
    prior chunks in that same file), and that accumulation is entirely
    local to one run_file() call. Results are reassembled in ORIGINAL
    file order (not completion order), so --report output stays
    deterministic across runs regardless of which file happened to
    finish first.

    isolate_workers routes each file through _run_file_in_worktree
    instead of calling run_file() directly on the shared root_dir — see
    git_ops/worktree.py's module docstring for what that buys and what
    it doesn't (there's no existing race this fixes; it's forward-looking
    isolation, opt-in because a worktree checkout per file isn't free)."""
    results: list[dict | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        future_to_index = {}
        for i, file_path in enumerate(files):
            siblings = [content for fp, content in file_contents.items() if fp != file_path]
            if options.isolate_workers:
                future = executor.submit(
                    _run_file_in_worktree, root_dir, file_path, options, siblings,
                )
            else:
                future = executor.submit(
                    run_file, file_path, dataclasses.replace(options, interactive=False),
                    False, siblings,
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


def run_repo(root_dir: str, options: RunOptions) -> list[dict]:
    global _last_target_test_result
    _last_target_test_result = None
    if options.interactive and options.workers > 1:
        # A synchronous input() prompt from N concurrent threads at once
        # is a genuinely broken UX (garbled, unpredictable terminal
        # output, no way to tell which file a prompt belongs to) — not
        # just noisier than the existing "console output WILL interleave"
        # warning for concurrent mode, actually unusable. Refuse clearly
        # rather than let it silently misbehave.
        raise ValueError("--interactive is not supported with --workers > 1 (concurrent prompts would garble the terminal). Run with --workers 1.")
    if options.isolate_workers and options.workers <= 1:
        raise ValueError("--isolate-workers only makes sense with --workers > 1.")
    if options.isolate_workers:
        from git_ops.worktree import is_git_repo
        if not is_git_repo(root_dir):
            raise ValueError(
                f"--isolate-workers requires {root_dir!r} to be a git repository "
                f"(needs `git worktree` to isolate each worker's checkout)."
            )
    if options.staged_only:
        from git_ops.staged import get_staged_files
        from git_ops.worktree import is_git_repo
        if not is_git_repo(root_dir):
            raise ValueError(f"--staged requires {root_dir!r} to be a git repository.")
        files = get_staged_files(root_dir)
        print(f"Found {len(files)} modernizable staged file(s) under {root_dir}")
    else:
        files = discover_files(root_dir)
        print(f"Found {len(files)} modernizable file(s) under {root_dir}")

    file_contents = _read_all_file_contents(files)

    baseline_test_result = None
    test_framework = None
    if options.run_target_tests:
        detected = detect_test_command(root_dir)
        if detected is None:
            print("\n--run-target-tests: no recognized test framework found at repo root — skipping.")
        else:
            test_framework, test_command = detected
            print(f"\n--run-target-tests: running {test_framework}'s existing test suite as a baseline "
                  f"(BEFORE modernization, on the untouched repo)...")
            baseline_test_result = run_test_command(root_dir, test_command)
            print(f"Baseline: {baseline_test_result['status']}")

    if options.workers > 1:
        print(
            f"Processing with {options.workers} concurrent workers. Files are "
            f"independent (only chunks WITHIN a file have ordering "
            f"dependencies via accumulated state), so this is safe — but "
            f"console output from different files WILL interleave. Use "
            f"--report for clean structured results instead of reading "
            f"the terminal for a concurrent run."
        )
        all_stats = _run_files_concurrently(files, options, file_contents, root_dir)
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
                    file_path, options,
                    standalone_pr=False, sibling_sources=siblings,
                )
            except Exception as e:
                print(f"ERROR processing {file_path}: {e}")
                stats = {"file_path": file_path, "error": str(e)}
            all_stats.append(stats)

    if options.run_target_tests and test_framework is not None:
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

    if options.open_pr:
        pr_url = _open_combined_pr(root_dir, all_stats)
        if pr_url:
            print(f"Opened combined PR: {pr_url}")
            for s in all_stats:
                if s.get("output_path"):
                    s["pr_url"] = pr_url

    return all_stats


def _scan_mtimes(root_dir: str) -> dict[str, float]:
    """{file_path: mtime} for every modernizable file discover_files()
    finds — the entire signal --watch uses to notice a change. Deliberately
    mtime-based polling, not a filesystem-events library (inotify/FSEvents/
    watchdog): this project already keeps its dependency list to what each
    feature strictly needs (see requirements.txt's per-line comments), and
    polling every few seconds is simple, has zero new dependencies, and
    works identically on every OS — the tradeoff (a few seconds of latency
    between an edit and this noticing it) is a fine price for that."""
    return {f: os.path.getmtime(f) for f in discover_files(root_dir) if os.path.isfile(f)}


def _diff_changed_files(old_mtimes: dict[str, float], new_mtimes: dict[str, float]) -> list[str]:
    """Files that are new OR whose mtime moved since the last scan.
    Deleted files (in old_mtimes but not new_mtimes) are silently dropped
    from tracking — nothing to modernize about a file that's gone."""
    return sorted(f for f, mtime in new_mtimes.items() if old_mtimes.get(f) != mtime)


def watch_run(
    root_dir: str, options: RunOptions,
    poll_interval_seconds: int = 5, initial_pass: bool = True,
    _max_polls: int | None = None,
) -> None:
    """Long-running incremental mode: modernize root_dir once (unless
    initial_pass=False), then poll for file changes and modernize ONLY
    what changed, forever — the "strangler fig" / gradual-migration
    pattern, where a team edits legacy code over time and wants each
    change swept up as it happens rather than re-running a full repo
    scan by hand. Each changed file goes through run_file() exactly like
    single-file mode (already-modern chunks are skipped for free by
    handler.already_modern(), so re-scanning an unrelated file that
    happened to have ITS mtime bumped by an unrelated git operation
    costs nothing beyond the chunking pass).

    Runs until Ctrl+C (KeyboardInterrupt), printed as a clean stop rather
    than a traceback. _max_polls is test-only plumbing — bounds the loop
    to a finite number of poll cycles so a test doesn't hang forever;
    real callers (main()) never pass it, so real usage is unaffected."""
    print(
        f"--watch: watching {root_dir} for changes, polling every "
        f"{poll_interval_seconds}s. Press Ctrl+C to stop."
    )

    if initial_pass:
        print("\n--watch: running an initial full pass before watching for changes...")
        run_repo(root_dir, dataclasses.replace(options, workers=1))

    known_mtimes = _scan_mtimes(root_dir)
    polls = 0
    try:
        while _max_polls is None or polls < _max_polls:
            time.sleep(poll_interval_seconds)
            polls += 1
            current_mtimes = _scan_mtimes(root_dir)
            changed = _diff_changed_files(known_mtimes, current_mtimes)
            known_mtimes = current_mtimes
            if not changed:
                continue

            print(f"\n--watch: {len(changed)} changed file(s) detected: {', '.join(changed)}")
            file_contents = _read_all_file_contents(list(current_mtimes.keys()))
            for file_path in changed:
                siblings = [content for fp, content in file_contents.items() if fp != file_path]
                try:
                    run_file(
                        file_path, options,
                        standalone_pr=options.open_pr, sibling_sources=siblings,
                    )
                except Exception as e:
                    print(f"--watch: ERROR processing {file_path}: {e}")
    except KeyboardInterrupt:
        print("\n--watch: stopped.")


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
            if c.get("used_deterministic_rule"):
                flags.append('<span class="badge badge-success">deterministic</span>')
            if c.get("punted"):
                flags.append('<span class="badge badge-neutral">punted</span>')
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
            f'{s.get("chunks_gave_up", 0)} gave up, '
            f'{s.get("chunks_punted", 0)} punted before attempting</p>'
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
    # flag on the command line always overrides both. --profile NAME (if
    # passed) is resolved FIRST, by peeking at sys.argv before the real
    # parser exists (argparse has no built-in way to make one flag's
    # value influence another flag's default), and its bundle of
    # settings-table keys is layered ON TOP of [settings] — so a profile
    # overrides the config file's defaults, but an explicit CLI flag
    # still overrides the profile, since argparse only ever changes a
    # `default=` when nothing was actually passed on the command line.
    _settings = dict(_config.get("settings", {}))
    _profile_parser = argparse.ArgumentParser(add_help=False)
    _profile_parser.add_argument("--profile", default=_settings.get("profile"))
    _profile_name = _profile_parser.parse_known_args()[0].profile
    if _profile_name:
        _profiles = load_profiles(_config)
        if _profile_name not in _profiles:
            _profile_parser.error(
                f"--profile '{_profile_name}' is not defined — built-in profiles: "
                f"{', '.join(sorted(config.BUILTIN_PROFILES))}. Add a [profiles.{_profile_name}] "
                f"table to .modernizer.toml to define a custom one."
            )
        _settings.update(_profiles[_profile_name])

    parser = argparse.ArgumentParser(description="Automated Code Modernization Engine")
    parser.add_argument("path", help="Path to a legacy source file, OR a directory to modernize recursively")
    parser.add_argument(
        "--profile", metavar="NAME", default=_profile_name,
        help="Apply a named bundle of settings as defaults (still overridable by any "
             "explicit flag below). Built-in: 'safe' (more iterations, punt-check, "
             "characterize, and regression tests all on — favors thoroughness over "
             "speed) and 'fast' (fewer iterations, all the optional verification/test-"
             "generation extras off — favors turnaround). Define your own by adding a "
             "[profiles.NAME] table to .modernizer.toml; it can override individual "
             "fields of a built-in profile of the same name, or introduce a new one.",
    )
    parser.add_argument("--pr", action="store_true", help="Open a GitHub PR for each modernized file")
    parser.add_argument(
        "--plan", action="store_true",
        help="Chunk the target and report scope + estimated LLM-call cost, WITHOUT calling the "
             "LLM or Docker sandbox at all — see what a real run would face before spending "
             "anything on it. Exits immediately after printing; every other flag except "
             "--max-iterations (which affects the cost estimate) and --report is ignored.",
    )
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
        "--sarif-report", metavar="PATH", default=_settings.get("sarif_report"),
        help="Write accumulated static-security-scan findings (see --help for how "
             "chunks get flagged) as a SARIF 2.1.0 file — the standard format GitHub "
             "Code Scanning / Advanced Security (and most other CI security dashboards) "
             "ingest directly, e.g. via github/codeql-action/upload-sarif in a workflow. "
             "Independent of --report/--report-html; pass any combination.",
    )
    parser.add_argument(
        "--workers", type=int, default=_settings.get("workers", 1),
        help="Process this many files concurrently in repo mode (default 1 = sequential). "
             "Each worker runs its own Docker containers and LLM calls — mind your machine's "
             "CPU/memory and Ollama's own concurrency limits before raising this.",
    )
    parser.add_argument(
        "--isolate-workers", action="store_true", default=_settings.get("isolate_workers", False),
        help="Requires --workers > 1 and root_dir to be a git repository. Each concurrent "
             "worker reads/writes inside its own `git worktree` checkout (isolated from every "
             "other worker and from a human editing the same tree mid-run) instead of the "
             "shared directory directly; output files are copied back afterward. Off by "
             "default — a worktree checkout per file isn't free, and nothing in the current "
             "pipeline actually races without it (see git_ops/worktree.py's module docstring).",
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
        "--characterize", action="store_true",
        default=_settings.get("characterize", False),
        help="For every chunk this run looks at (successful or given up on — NOT chunks "
             "skipped by --punt-check, which capture no probe data), write its ORIGINAL "
             "(pre-modernization) behavior — including quirks/bugs — into a standalone, "
             "durable characterization test file, so a chunk this project couldn't handle "
             "still leaves behind a safety net for a human (or a future attempt) to check "
             "against. Independent of --generate-regression-tests (which only covers "
             "successful chunks and embeds the MODERNIZED code) — use both to get both. "
             "New sibling file only (e.g. test_calc_characterization.py) — never overwrites "
             "anything. Supported for python/javascript/typescript/php; a no-op for cpp/java.",
    )
    parser.add_argument(
        "--staged", action="store_true", default=_settings.get("staged", False),
        help="Repo mode only: process ONLY files currently staged for commit (`git add`), "
             "instead of every modernizable file under `path`. The event-driven, pre-commit-"
             "hook-shaped complement to --watch (triggered by the commit itself, not a "
             "background poll loop) — typical use: a .git/hooks/pre-commit that runs "
             "`code-modernizer --staged .` before every commit. `path` must be a git "
             "repository; cross-file context grounding (sibling signatures/types) only sees "
             "OTHER staged files, not the whole repo, in this mode.",
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
    parser.add_argument(
        "--recipe", metavar="NAME", default=_settings.get("recipe"),
        help="Name of a [recipes.NAME] table in .modernizer.toml to scope this run — its "
             "`instruction` string is appended to the base system prompt for every chunk, e.g. "
             "'only convert callback-style functions to async/await, leave everything else "
             "as-is.' Without this, every chunk gets the same generic 'modernize this' prompt.",
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Long-running incremental mode: run an initial full pass over `path` (a "
             "directory), then poll for file changes and modernize ONLY what changed, "
             "forever — for gradual/'strangler fig' migration where you want each edit "
             "swept up as it happens. Runs until Ctrl+C. Not compatible with --plan.",
    )
    parser.add_argument(
        "--watch-interval", type=int, default=_settings.get("watch_interval_seconds", 5),
        help="Seconds between change-detection polls in --watch mode (default 5).",
    )
    parser.add_argument(
        "--punt-check", action="store_true", default=_settings.get("punt_check", False),
        help="Before attempting each chunk, ask the model to self-assess whether it's "
             "confident it can modernize this specific chunk correctly. If it isn't, the "
             "chunk is skipped BEFORE any rewrite attempt (no best-of-N generation, no "
             "sandbox verification) — costs one extra LLM call per chunk to potentially save "
             "a much larger one on chunks the model already doubts. Punted chunks are NOT "
             "counted toward a language's track record (see track_record.py) — a chunk never "
             "attempted isn't a failed attempt.",
    )
    args = parser.parse_args()

    _recipe_instruction = None
    if args.recipe:
        _recipes = load_recipes(_config)
        if args.recipe not in _recipes:
            parser.error(
                f"--recipe '{args.recipe}' is not defined — add a [recipes.{args.recipe}] "
                f"table with an `instruction` string to .modernizer.toml. "
                f"Available: {', '.join(sorted(_recipes)) or '(none defined)'}"
            )
        _recipe_instruction = _recipes[args.recipe]

    _options = RunOptions(
        open_pr=args.pr,
        max_iterations=args.max_iterations,
        workers=args.workers,
        run_target_tests=args.run_target_tests,
        generate_regression_tests=args.generate_regression_tests,
        interactive=args.interactive,
        recipe_instruction=_recipe_instruction,
        punt_check_enabled=args.punt_check,
        characterize=args.characterize,
        isolate_workers=args.isolate_workers,
        staged_only=args.staged,
    )

    if args.plan and args.watch:
        parser.error("--plan and --watch are not compatible — --plan makes one static estimate, --watch runs forever.")
    if args.staged and args.watch:
        parser.error("--staged and --watch are not compatible — --staged is a one-shot pre-commit-hook mode, --watch runs forever.")
    if args.staged and not os.path.isdir(args.path):
        parser.error("--staged requires `path` to be a directory (a git repository).")

    if args.plan:
        # Deliberately short-circuits before touching llm_budget, Docker,
        # or any of the run_file/run_repo machinery below — --plan's
        # entire point is a zero-cost, zero-side-effect look at scope.
        plans = plan_run(args.path, args.max_iterations)
        print_plan(plans)
        if args.report:
            import json
            with open(args.report, "w") as f:
                json.dump({"mode": "plan", "plans": plans}, f, indent=2)
            print(f"\nWrote plan report to {args.report}")
        return

    if args.watch:
        if not os.path.isdir(args.path):
            parser.error("--watch requires `path` to be a directory (it watches for file changes over time).")
        llm_budget.reset(max_calls=args.max_llm_calls)
        watch_run(args.path, _options, poll_interval_seconds=args.watch_interval)
        return

    if args.interactive and args.workers > 1:
        parser.error("--interactive is not supported with --workers > 1 (concurrent prompts would garble the terminal)")

    # Reset ONCE here, at the true top-level entry point — never inside
    # run_file/run_repo themselves, or a repo-wide budget would silently
    # reset per file instead of accumulating across the whole run (see
    # LLMBudget.reset's docstring).
    llm_budget.reset(max_calls=args.max_llm_calls)

    if os.path.isdir(args.path):
        results = run_repo(args.path, _options)
        mode = "repo"
    else:
        results = [run_file(args.path, _options)]
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
    if args.sarif_report:
        write_sarif_report(args.sarif_report, results)
        print(f"Wrote SARIF security report to {args.sarif_report}")


if __name__ == "__main__":
    main()
