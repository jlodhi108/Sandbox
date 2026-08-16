from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes import (
    refactorer_node, verifier_node, fallback_node, assess_risk, scan_security,
    generate_probes, wrap_call_as_probe, check_mutation_confidence,
)
from agents.review_graph import start_review
from languages import get_handler_by_name
from sandbox.verifier import verify

MAX_PROBES_PER_CHUNK = 3
MAX_REAL_CALL_SITE_PROBES = 2


def router(state: AgentState) -> str:
    if state["status"] == "success":
        return "success"
    if state["iteration_count"] >= state["max_iterations"]:
        return "give_up"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("refactorer", refactorer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("refactorer")
    graph.add_edge("refactorer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        router,
        {
            "success": END,
            "retry": "refactorer",
            "give_up": "fallback",
        },
    )
    graph.add_edge("fallback", END)

    return graph.compile()


def _capture_baseline_stdout(language: str, full_source: bytes) -> str | None:
    """Run the file as-is (before this chunk's modernization) once, so
    verifier_node can later confirm the modernized version produces
    IDENTICAL output — "compiles and exits 0" is not the same as
    "behaves the same," and we've caught real bugs (silently changed
    output) that only a real behavioral diff would catch. Returns None
    if the original doesn't run cleanly — nothing to compare against, so
    the check is skipped rather than blocking all modernization on a
    baseline that was already broken."""
    handler = get_handler_by_name(language)
    baseline = verify(full_source.decode("utf-8"), handler.sandbox_filename, handler.run_command())
    return baseline["stdout"] if baseline["status"] == "success" else None


def _find_real_call_site_probes(
    handler, full_source: bytes, original_code: str, sibling_sources: list[bytes]
) -> list[str]:
    """Real calls to this function, found via Tree-sitter across the
    current file and (in repo mode) every sibling file — strictly
    stronger probe input than an LLM guess: it proves the function still
    works for how it's REALLY called, not how a model imagines it might
    be called. Capped and de-duplicated; a bare call expression here
    still needs wrapping in a print/echo before it's a runnable probe,
    done by the caller."""
    function_name = handler.extract_function_name(original_code)
    if not function_name:
        return []

    sites: list[str] = []
    seen = set()
    for source in [full_source] + sibling_sources:
        for site in handler.find_call_sites(function_name, source):
            if site not in seen:
                seen.add(site)
                sites.append(site)
            if len(sites) >= MAX_REAL_CALL_SITE_PROBES:
                return sites
    return sites


def _capture_function_probes(
    language: str,
    full_source: bytes,
    original_code: str,
    sibling_sources: list[bytes] | None = None,
) -> list[dict]:
    """Build a list of {snippet, baseline_stdout} probes for this chunk's
    function: real call sites (see _find_real_call_site_probes) PLUS
    LLM-synthesized diverse examples filling any remaining slots — a
    real call site proves one real usage works, but says nothing about
    edge cases (empty/zero/boundary) that usage happens not to exercise,
    so both sources matter. Each candidate is verified against the
    UNCHANGED original before being trusted; ones that don't run cleanly
    (bad guess, or a real call site that referenced a local variable not
    available at file scope where the probe gets appended) are dropped
    rather than blocking modernization. Only attempted for languages
    where appending a probe is safe (see LanguageHandler.
    supports_function_probe)."""
    handler = get_handler_by_name(language)
    if not handler.supports_function_probe:
        return []

    real_sites = _find_real_call_site_probes(handler, full_source, original_code, sibling_sources or [])
    candidates = [wrap_call_as_probe(language, site) for site in real_sites]

    remaining = max(0, MAX_PROBES_PER_CHUNK - len(candidates))
    if remaining:
        # De-dup against the real-site-derived candidates: the model
        # sometimes independently synthesizes the exact same example a
        # real call site already produced (confirmed live — it picked
        # the same call main() already makes), which would otherwise
        # waste a sandbox round-trip checking the same input twice.
        synthesized = [s for s in generate_probes(language, original_code, count=remaining) if s not in candidates]
        candidates.extend(synthesized)

    if not candidates:
        print("    (no probes available for this chunk — skipping probe check)")
        return []

    probes = []
    for snippet in candidates:
        probe_candidate = full_source + b"\n" + snippet.encode("utf-8") + b"\n"
        result = verify(probe_candidate.decode("utf-8"), handler.sandbox_filename, handler.run_command())
        if result["status"] != "success":
            print(f"    (probe {snippet!r} failed against the ORIGINAL function — dropping)")
            continue
        print(f"    Probe: {snippet}  ->  baseline: {result['stdout']!r}")
        probes.append({"snippet": snippet, "baseline_stdout": result["stdout"]})
    return probes


def modernize(
    language: str,
    full_source: bytes,
    chunk_start: int,
    chunk_end: int,
    max_iterations: int = 5,
    sibling_sources: list[bytes] | None = None,
    interactive: bool = False,
) -> AgentState:
    app = build_graph()
    original_code = full_source[chunk_start:chunk_end].decode("utf-8")
    baseline_stdout = _capture_baseline_stdout(language, full_source)
    probes = _capture_function_probes(language, full_source, original_code, sibling_sources)
    initial_state: AgentState = {
        "language": language,
        "full_source": full_source,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "original_code": original_code,
        "modernized_code": "",
        "required_imports": [],
        "candidate_codes": [],
        "baseline_stdout": baseline_stdout,
        "probes": probes,
        "used_escalation": False,
        "risk_flag": False,
        "risk_reason": "",
        "security_flag": False,
        "security_findings": [],
        "mutation_confidence_flag": False,
        "mutation_confidence_reason": "",
        "review_thread_id": None,
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
        "max_iterations": max_iterations,
    }
    final_state = app.invoke(initial_state)

    # Risk assessment, security scan, and mutation-confidence check all
    # run once, after the graph is done, only for chunks that actually
    # succeeded — no point spending extra cycles critiquing a change
    # that already got rejected.
    if final_state["status"] == "success":
        risk_flag, risk_reason = assess_risk(final_state["modernized_code"], final_state["used_escalation"])
        security_flag, security_findings = scan_security(language, final_state["modernized_code"])
        handler = get_handler_by_name(language)
        mutation_confidence_flag, mutation_confidence_reason = check_mutation_confidence(
            handler, final_state, final_state["modernized_code"], final_state["required_imports"]
        )
        final_state = {
            **final_state,
            "risk_flag": risk_flag,
            "risk_reason": risk_reason,
            "security_flag": security_flag,
            "security_findings": security_findings,
            "mutation_confidence_flag": mutation_confidence_flag,
            "mutation_confidence_reason": mutation_confidence_reason,
        }

        # interactive=False (default): completely unchanged from before
        # this feature existed — flags are surfaced, nothing pauses.
        # interactive=True: hand the (already-computed) flags to the
        # SEPARATE review graph (agents/review_graph.py) rather than
        # blocking here — a flagged chunk comes back with status
        # "awaiting_review" and a review_thread_id instead of "success",
        # for the caller (main.py's --interactive prompt, or
        # mcp_server.py's resume_chunk_review tool) to resolve.
        if interactive:
            review = start_review({
                "risk_flag": risk_flag,
                "risk_reason": risk_reason,
                "security_flag": security_flag,
                "security_findings": security_findings,
                "mutation_confidence_flag": mutation_confidence_flag,
                "mutation_confidence_reason": mutation_confidence_reason,
            })
            if review["status"] == "awaiting_review":
                final_state = {
                    **final_state,
                    "status": "awaiting_review",
                    "review_thread_id": review["thread_id"],
                }

    return final_state


if __name__ == "__main__":
    from languages import get_handler

    with open("legacy_samples/legacy.cpp", "rb") as f:
        source = f.read()

    # demo: modernize just the last chunk in isolation
    handler = get_handler("legacy_samples/legacy.cpp")
    chunks = handler.chunk(source)
    last = chunks[-1]

    final_state = modernize(handler.name, source, last.start_byte, last.end_byte)
    print("--- STATUS:", final_state["status"], "---")
    print("--- ITERATIONS:", final_state["iteration_count"], "---")
    print(final_state["modernized_code"])
