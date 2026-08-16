from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes import refactorer_node, verifier_node, fallback_node, assess_risk, generate_probe
from languages import get_handler_by_name
from sandbox.verifier import verify


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


def _capture_function_probe(
    language: str, full_source: bytes, original_code: str
) -> tuple[str | None, str | None]:
    """Generate a probe for this chunk's function and run it once against
    the ORIGINAL (pre-modernization) file to get baseline output. Closes
    the gap the whole-file baseline check has: that check only covers
    whatever the file's own entry point exercises, so a function nothing
    calls yet gets zero protection from it. Only attempted for languages
    where appending a probe is safe (see LanguageHandler.
    supports_function_probe). Returns (None, None) on any failure —
    probe generation is best-effort and must never block modernization
    that the other checks already prove is safe."""
    handler = get_handler_by_name(language)
    if not handler.supports_function_probe:
        return None, None

    probe_snippet = generate_probe(language, original_code)
    if probe_snippet is None:
        print("    (no probe generated for this chunk — skipping probe check)")
        return None, None

    probe_candidate = full_source + b"\n" + probe_snippet.encode("utf-8") + b"\n"
    result = verify(probe_candidate.decode("utf-8"), handler.sandbox_filename, handler.run_command())
    if result["status"] != "success":
        # Probe doesn't even run cleanly against the UNCHANGED original —
        # untrustworthy (bad argument choice, wrong assumptions about the
        # function). Skip rather than block on a broken probe.
        print(f"    (probe {probe_snippet!r} failed against the ORIGINAL function — skipping probe check)")
        return None, None

    print(f"    Probe generated: {probe_snippet}  ->  baseline: {result['stdout']!r}")
    return probe_snippet, result["stdout"]


def modernize(
    language: str,
    full_source: bytes,
    chunk_start: int,
    chunk_end: int,
    max_iterations: int = 5,
) -> AgentState:
    app = build_graph()
    original_code = full_source[chunk_start:chunk_end].decode("utf-8")
    baseline_stdout = _capture_baseline_stdout(language, full_source)
    probe_snippet, probe_baseline_stdout = _capture_function_probe(language, full_source, original_code)
    initial_state: AgentState = {
        "language": language,
        "full_source": full_source,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "original_code": original_code,
        "modernized_code": "",
        "required_imports": [],
        "baseline_stdout": baseline_stdout,
        "probe_snippet": probe_snippet,
        "probe_baseline_stdout": probe_baseline_stdout,
        "used_escalation": False,
        "risk_flag": False,
        "risk_reason": "",
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
        "max_iterations": max_iterations,
    }
    final_state = app.invoke(initial_state)

    # Risk assessment runs once, after the graph is done, only for chunks
    # that actually succeeded — no point spending an LLM call critiquing
    # a change that already got rejected.
    if final_state["status"] == "success":
        risk_flag, risk_reason = assess_risk(final_state["modernized_code"])
        final_state = {**final_state, "risk_flag": risk_flag, "risk_reason": risk_reason}

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
