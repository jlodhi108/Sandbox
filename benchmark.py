"""Self-benchmark: run the FULL pipeline (real LLM calls, real sandbox
verification) against a fixed set of legacy code fixtures and report a
scorecard — chunks succeeded/total per language, success rate, timing.

This exists as a repeatable, comparable-over-time trust signal for how
well this project's own verification pipeline actually performs, rather
than asking anyone to just take "it works" on faith — directly aimed at
the "AI coding tools are broadly distrusted" gap the 2026 Stack Overflow
Developer Survey and multiple "vibe coding" retrospectives keep
surfacing: an outside user can run this themselves, or diff two runs of
it (before/after a prompt tweak, a model swap, a threshold change) to
see whether a change actually helped or quietly regressed something.

Deliberately SEPARATE from track_record.py's per-run history (which
gates auto-PR eligibility for REAL user runs against REAL repos): this
benchmarks against FIXED, known fixtures (legacy_samples/ by default)
purely for measuring pipeline quality itself, and must never feed back
into or be confused with the auto-PR trust gate for someone's actual
codebase — this script never calls record_run/save_history at all.

NOT a CI-unit-test — it makes real Ollama calls and real Docker sandbox
verifications, exactly like a normal `main.py` run, so it's slow,
non-deterministic, and costs whatever LLM calls a real run costs. Run it
manually, or via the separate, workflow_dispatch/schedule-only
.github/workflows/benchmark.yml (never on every push/PR — that would
slow down and destabilize ordinary CI with real model output)."""
import argparse
import json
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from config import load_config, apply_config_to_environment

load_dotenv()
_config = load_config()
apply_config_to_environment(_config)

from main import run_file, discover_files, RunOptions
from agents.nodes import llm_budget

DEFAULT_SAMPLES_DIR = "legacy_samples"


def run_benchmark(samples_dir: str, max_iterations: int = 5, max_llm_calls: int | None = None) -> dict:
    """Runs run_file() against every modernizable fixture under
    samples_dir (same discovery/filtering as repo mode — see
    main.py:discover_files) and aggregates the results into a
    per-language + overall scorecard. Does NOT open PRs, generate
    regression tests, or touch track_record.py — pure measurement."""
    llm_budget.reset(max_calls=max_llm_calls)
    files = discover_files(samples_dir)

    options = RunOptions(open_pr=False, max_iterations=max_iterations)
    start = time.time()
    file_results = []
    for file_path in files:
        try:
            stats = run_file(file_path, options)
        except Exception as e:
            stats = {"file_path": file_path, "error": str(e)}
        file_results.append(stats)
    duration = time.time() - start

    by_language: dict[str, dict] = {}
    for s in file_results:
        if "error" in s:
            continue
        entry = by_language.setdefault(s["language"], {
            "files": 0, "chunks_total": 0, "chunks_succeeded": 0,
            "chunks_already_modern": 0, "chunks_gave_up": 0, "chunks_punted": 0,
        })
        entry["files"] += 1
        entry["chunks_total"] += s.get("chunks_total", 0)
        entry["chunks_succeeded"] += s.get("chunks_succeeded", 0)
        entry["chunks_already_modern"] += s.get("chunks_already_modern", 0)
        entry["chunks_gave_up"] += s.get("chunks_gave_up", 0)
        entry["chunks_punted"] += s.get("chunks_punted", 0)

    total_chunks = sum(v["chunks_total"] for v in by_language.values())
    total_succeeded = sum(v["chunks_succeeded"] for v in by_language.values())
    # "Attempted" excludes already-modern and punted chunks, same
    # definition track_record.py uses — a chunk the pipeline never
    # actually tried isn't evidence about how well it performs when it
    # DOES try.
    total_attempted = sum(v["chunks_succeeded"] + v["chunks_gave_up"] for v in by_language.values())

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples_dir": samples_dir,
        "max_iterations": max_iterations,
        "duration_seconds": duration,
        "llm_usage": llm_budget.summary(),
        "by_language": by_language,
        "overall": {
            "chunks_total": total_chunks,
            "chunks_succeeded": total_succeeded,
            "chunks_attempted": total_attempted,
            "success_rate_of_attempted": (total_succeeded / total_attempted) if total_attempted else None,
        },
        "file_results": file_results,
    }


def print_scorecard(benchmark: dict) -> None:
    print(f"\n{'=' * 64}\nSelf-benchmark scorecard — {benchmark['generated_at']}\n{'=' * 64}")
    for lang, stats in sorted(benchmark["by_language"].items()):
        attempted = stats["chunks_succeeded"] + stats["chunks_gave_up"]
        rate = f"{stats['chunks_succeeded'] / attempted:.0%}" if attempted else "n/a"
        print(
            f"  {lang:12s} {stats['chunks_succeeded']}/{attempted} attempted succeeded ({rate})"
            f" | {stats['chunks_already_modern']} already modern"
            f" | {stats['chunks_punted']} punted"
            f" | {stats['files']} file(s)"
        )
    overall = benchmark["overall"]
    rate = (
        f"{overall['success_rate_of_attempted']:.0%}"
        if overall["success_rate_of_attempted"] is not None else "n/a"
    )
    print(
        f"\nOverall: {overall['chunks_succeeded']}/{overall['chunks_total']} chunk(s) succeeded"
        f" — success rate of attempted chunks: {rate}"
    )
    usage = benchmark["llm_usage"]
    print(f"LLM calls: {usage['total_calls']}  |  Duration: {benchmark['duration_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Self-benchmark: run the real pipeline against fixed legacy-code fixtures and report a scorecard.",
    )
    parser.add_argument(
        "--samples-dir", default=DEFAULT_SAMPLES_DIR,
        help=f"Directory of fixture files to benchmark against (default: {DEFAULT_SAMPLES_DIR})",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-llm-calls", type=int, default=None, help="Hard ceiling on total LLM calls for this benchmark run")
    parser.add_argument("--report", metavar="PATH", help="Write the full benchmark JSON to this path")
    args = parser.parse_args()

    benchmark = run_benchmark(args.samples_dir, args.max_iterations, args.max_llm_calls)
    print_scorecard(benchmark)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"\nWrote benchmark report to {args.report}")


if __name__ == "__main__":
    main()
