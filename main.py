import argparse
import os
import time
from dotenv import load_dotenv

# Must run before importing agents.graph — agents.nodes reads
# ESCALATION_MODEL from the environment at import time, so anything
# configured only in .env (not already exported in the shell) would
# silently never take effect if load_dotenv() ran after that import.
load_dotenv()

from languages import get_handler
from agents.graph import modernize
from sandbox.verifier import verify
from git_ops.pr import open_modernization_pr

# Common vendor/build/VCS directories to never descend into in repo mode —
# nothing in here is source you'd want modernized, and node_modules/.git
# in particular can be enormous.
_EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", "out", ".next", "target", ".pytest_cache",
}


def run_file(file_path: str, open_pr: bool, max_iterations: int) -> dict:
    """Modernize one file. Returns a stats dict (also the data source for
    --report) rather than just printing, so repo mode can aggregate
    results across many files without re-parsing terminal output."""
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
            max_iterations=max_iterations,
        )
        print(f"status={final_state['status']} iterations={final_state['iteration_count']}")

        chunk_detail = {
            "kind": chunk.kind,
            "start_byte": chunk.start_byte,
            "status": final_state["status"],
            "iterations": final_state["iteration_count"],
            "used_escalation": final_state.get("used_escalation", False),
            "had_probe": final_state.get("probe_snippet") is not None,
            "risk_flag": final_state.get("risk_flag", False),
            "duration_seconds": time.time() - chunk_start_time,
        }
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

    if open_pr:
        if not final_check_passed:
            print("\nRefusing to open a PR: final behavioral check failed. "
                  "Inspect the output file manually before proposing it.")
            stats["duration_seconds"] = time.time() - start_time
            return stats
        risk_flagged = stats["risk_flagged"]
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
                + (
                    f"**{len(risk_flagged)} chunk(s) flagged for manual review** "
                    f"(touch I/O, global state, randomness, or timing — the "
                    f"automated stdout check can't fully prove these are safe):\n"
                    + "\n".join(f"- {r['kind']} at byte {r['start_byte']}: {r['reason']}" for r in risk_flagged)
                    if risk_flagged else "No chunks flagged as higher-risk."
                )
            ),
        )
        stats["pr_url"] = url
        print(f"Opened PR: {url}")

    stats["duration_seconds"] = time.time() - start_time
    return stats


def discover_files(root_dir: str) -> list[str]:
    """Walk root_dir, returning every file with a registered language
    extension, skipping common vendor/build/VCS directories."""
    from languages import get_handler

    found = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDED_DIRS]
        for filename in filenames:
            if ".modernized." in filename:
                # This tool's own output — same extension as its input,
                # so get_handler() would happily match it too. Without
                # this, re-running repo mode would treat every previous
                # run's output as new input and modernize it again.
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                get_handler(full_path)
            except ValueError:
                continue
            found.append(full_path)
    return sorted(found)


def run_repo(root_dir: str, open_pr: bool, max_iterations: int) -> list[dict]:
    files = discover_files(root_dir)
    print(f"Found {len(files)} modernizable file(s) under {root_dir}")

    all_stats = []
    for i, file_path in enumerate(files, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(files)}] {file_path}\n{'=' * 60}")
        try:
            stats = run_file(file_path, open_pr, max_iterations)
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            stats = {"file_path": file_path, "error": str(e)}
        all_stats.append(stats)

    print(f"\n{'=' * 60}\nRepo summary ({len(files)} file(s))\n{'=' * 60}")
    total_succeeded = sum(s.get("chunks_succeeded", 0) for s in all_stats)
    total_chunks = sum(s.get("chunks_total", 0) for s in all_stats)
    total_risk = sum(len(s.get("risk_flagged", [])) for s in all_stats)
    files_written = sum(1 for s in all_stats if s.get("output_path"))
    files_prs = sum(1 for s in all_stats if s.get("pr_url"))
    print(f"Chunks modernized: {total_succeeded}/{total_chunks}")
    print(f"Files with output written: {files_written}/{len(files)}")
    print(f"Chunks flagged for manual review: {total_risk}")
    if open_pr:
        print(f"PRs opened: {files_prs}")

    return all_stats


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
    parser = argparse.ArgumentParser(description="Automated Code Modernization Engine")
    parser.add_argument("path", help="Path to a legacy source file, OR a directory to modernize recursively")
    parser.add_argument("--pr", action="store_true", help="Open a GitHub PR for each modernized file")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--report", metavar="PATH", help="Write a structured JSON run report to this path")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        results = run_repo(args.path, args.pr, args.max_iterations)
        mode = "repo"
    else:
        results = [run_file(args.path, args.pr, args.max_iterations)]
        mode = "file"

    if args.report:
        write_report(args.report, mode, results)
