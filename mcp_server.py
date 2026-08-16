"""MCP server exposing code-modernizer's pipeline as callable tools, so
an MCP client (Claude Code, Cursor, Claude Desktop, etc.) can drive a
modernization run directly instead of a human invoking main.py by hand.

Run with `python mcp_server.py` (stdio transport — what local MCP
clients expect) or `fastmcp run mcp_server.py` for other transports.
Requires Ollama running locally and Docker available, same as the CLI.

`import main` (rather than reimplementing its setup) deliberately reuses
main.py's own load_dotenv() -> load_config() -> apply_config_to_environment()
sequence, which MUST run in that order before agents.graph/sandbox.verifier
are imported (both read env vars at import time — see main.py's own
comment on this). That sequence lives inside main.py at module level, so
importing it here runs it exactly once, correctly ordered, without a
second copy to keep in sync. The `if __name__ == "__main__":` guard
means the CLI's argparse/run block never fires from this import.

file_path/root_dir arguments to the tools below should be ABSOLUTE
paths. This server chdir()s to its own directory below so config/
history files (.env, .modernizer.toml, .modernizer_history.json — all
looked up by main.py/config.py/track_record.py via relative,
cwd-dependent paths) resolve correctly regardless of what directory the
MCP CLIENT happens to launch it from — most client configs (Claude
Code, Cursor) specify a command + args, not a cwd, so without this the
server's actual working directory would be whatever the client
application itself started from, silently missing this project's own
config/history. That same chdir means a RELATIVE file_path/root_dir
would resolve against this project's own directory, not wherever the
calling agent is actually working — almost never what's wanted, since
this tool exists to modernize code in OTHER projects.
"""
import contextlib
import os
import sys

# Anchor to this file's own directory BEFORE importing main (which loads
# .env/.modernizer.toml/.modernizer_history.json using paths relative to
# the process's cwd at that moment) — see module docstring above.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP

import main
from track_record import record_run, save_history, is_eligible

mcp = FastMCP(
    "code-modernizer",
    instructions=(
        "Modernizes legacy C++/Python/JavaScript/TypeScript/Java/PHP code "
        "using a local LLM with sandboxed, execution-grounded verification: "
        "every candidate rewrite is compiled/run in an isolated, "
        "network-disabled Docker container and checked against the "
        "original's behavior before being accepted. Nothing is ever "
        "written back over the original file — output goes to a sibling "
        "*.modernized.* file for review."
    ),
)


@contextlib.contextmanager
def _protect_stdout():
    """main.py/agents/*.py call print() extensively for human-readable
    CLI progress output. Under stdio transport (what this server uses),
    stdout IS the JSON-RPC protocol channel — confirmed by direct test
    that a print()-heavy tool call reliably corrupts the client's
    message stream without this (repeated pydantic ValidationErrors on
    the client side, one per stray line). Redirecting to stderr for the
    call's duration keeps the existing pipeline code untouched — no need
    to hunt down and silence prints scattered across the whole
    codebase — while guaranteeing the protocol channel itself stays
    clean. (Residual caveat: sys.stdout is process-global, not
    thread-local, so this could in principle also swallow a stdout write
    from a genuinely CONCURRENT second tool call mid-flight; not a
    concern for the single-tool-call-at-a-time usage this server is
    built for.)"""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _update_track_record(results: list[dict]) -> None:
    """Mirrors main.py's __main__ block: fold this run's results into
    the persisted track record, whether or not a PR was requested — an
    MCP-driven run went through the identical verification pipeline as
    a CLI-driven one and should count toward eligibility the same way.
    Updates main._history in place (module-global) so a SECOND tool call
    later in this same server process sees the accumulated total, not a
    stale snapshot from server startup."""
    updated = main._history
    for s in results:
        language = s.get("language")
        if not language:
            continue
        attempted = s.get("chunks_succeeded", 0) + s.get("chunks_gave_up", 0)
        if attempted == 0:
            continue
        updated = record_run(updated, language, s.get("chunks_succeeded", 0), attempted)
    if updated != main._history:
        save_history(updated)
        main._history = updated


@mcp.tool
def modernize_file(file_path: str, max_iterations: int = 5, open_pr: bool = False) -> dict:
    """Modernize one legacy source file (C++/Python/JS/TS/Java/PHP,
    detected from its extension). file_path should be an ABSOLUTE path —
    this server runs from its own project directory, so a relative path
    would resolve there instead of wherever you're actually working.
    Writes the result to
    <name>.modernized.<ext> next to the original — the original file is
    never overwritten. Each function/method is independently rewritten
    by a local LLM and verified in an isolated Docker sandbox (compiled/
    run, output compared against the original) before being accepted;
    anything that fails verification is left unchanged in the output.

    open_pr=True additionally attempts to open a GitHub PR with the
    changes, but ONLY if the whole-file behavioral check passes AND this
    language already has a proven track record in THIS project (see
    check_track_record) — otherwise the PR is refused and this still
    returns normally with pr_url left as None. Requires GITHUB_TOKEN/
    GITHUB_REPO configured in the environment. Leave this False to just
    generate the modernized file for your own review first.

    Returns a stats dict: chunks_total/succeeded/already_modern/gave_up,
    output_path, final_check_passed, pr_url, and a chunk_details list
    (per-chunk status, diff, risk/security flags)."""
    with _protect_stdout():
        stats = main.run_file(file_path, open_pr, max_iterations)
        _update_track_record([stats])
    return stats


@mcp.tool
def modernize_repo(root_dir: str, max_iterations: int = 5, workers: int = 1, open_pr: bool = False) -> dict:
    """Modernize every supported source file under root_dir (recursive,
    .gitignore-aware, skips node_modules/.git/build output dirs etc.).
    root_dir should be an ABSOLUTE path — this server runs from its own
    project directory, so a relative path would resolve there instead
    of wherever you're actually working. Same per-file verification as
    modernize_file, run across the whole tree; each file's output goes
    to its own sibling <name>.modernized.<ext>.

    open_pr=True opens at most ONE combined PR containing every file
    that passed its final behavioral check AND whose language is
    track-record-eligible — a mixed-language repo commonly has some
    languages proven and others brand new to this tool, so eligibility
    is checked per file's own language, not once for the whole repo.

    workers > 1 processes files concurrently (each spins up its own
    Docker containers and LLM calls — mind local CPU/memory and Ollama's
    own concurrency limits before raising this).

    Returns {"files": [...one stats dict per file, see modernize_file...],
    "summary": {...aggregate totals across the run...}}."""
    with _protect_stdout():
        results = main.run_repo(root_dir, open_pr, max_iterations, workers)
        _update_track_record(results)
    return {
        "files": results,
        "summary": {
            "files_processed": len(results),
            "chunks_succeeded": sum(r.get("chunks_succeeded", 0) for r in results),
            "chunks_total": sum(r.get("chunks_total", 0) for r in results),
            "files_with_output": sum(1 for r in results if r.get("output_path")),
            "chunks_flagged_risk": sum(len(r.get("risk_flagged", [])) for r in results),
            "chunks_flagged_security": sum(len(r.get("security_flagged", [])) for r in results),
        },
    }


@mcp.tool
def check_track_record(language: str) -> dict:
    """Check whether `language` (e.g. "python", "javascript",
    "typescript", "cpp", "java", "php") currently has a proven-enough
    track record in THIS project to be eligible for open_pr=True on
    modernize_file/modernize_repo — gated on a minimum number of
    attempted chunks and a minimum success rate (both configurable via
    .modernizer.toml's [autonomy] table). A language with no history at
    all is always ineligible (fails closed) rather than silently
    allowed through."""
    ok, reason = is_eligible(
        language, main._history, main._MIN_TRACK_RECORD_CHUNKS, main._MIN_SUCCESS_RATE,
    )
    entry = main._history.get(language, {"chunks_succeeded": 0, "chunks_attempted": 0})
    return {
        "language": language,
        "eligible_for_auto_pr": ok,
        "reason": reason,
        "chunks_succeeded": entry["chunks_succeeded"],
        "chunks_attempted": entry["chunks_attempted"],
    }


@mcp.tool
def list_track_record() -> dict:
    """Full per-language track record accumulated across every past run
    in THIS project (backed by .modernizer_history.json) — succeeded and
    attempted chunk counts per language. Use check_track_record for a
    single language's pass/fail auto-PR eligibility verdict; this is the
    raw counts across all of them at once."""
    return dict(main._history)


if __name__ == "__main__":
    mcp.run(transport="stdio")
