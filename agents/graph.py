import re

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes import (
    refactorer_node, verifier_node, fallback_node, assess_risk, scan_security,
    generate_probes, wrap_call_as_probe, check_mutation_confidence, assess_punt,
)
from agents.review_graph import start_review
from languages import get_handler_by_name
from sandbox.verifier import verify

MAX_PROBES_PER_CHUNK = 3
MAX_REAL_CALL_SITE_PROBES = 2
MAX_CONTEXT_SIGNATURES = 20
MAX_CONTEXT_SIBLING_FILES = 5
MAX_TYPE_DEFINITIONS = 5


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


def _extract_context_signatures(
    handler, full_source: bytes, chunk_start: int, chunk_end: int, sibling_sources: list[bytes] | None,
) -> list[str]:
    """One-line signatures of every OTHER function/method in the current
    file, plus a bounded sample from sibling files (repo mode only) —
    grounding for the refactor prompt so the model can reference the
    codebase's ACTUAL names/types with confidence instead of guessing.
    This is the scoped-down, same-language version of the "static
    analysis first" idea from AlphaTrans/LegacyTranslate-style
    architectures: those build a full cross-language project skeleton
    because they translate BETWEEN languages and need to resolve types
    across that boundary; this project modernizes Python-to-Python,
    JS-to-JS, etc., so a short list of real signatures already visible
    via the chunking this project already does is the right-sized
    version of the same idea, not a new dependency-resolution system.

    Deliberately reuses handler.chunk() (the same tree-sitter chunking
    every file already goes through) rather than a new per-language
    signature-extraction query — each chunk's FIRST LINE is taken as
    its "signature", which is exact for single-line signatures (the
    overwhelming majority of real code) and merely truncated-looking
    (not wrong, not crash-prone) for multi-line ones. This is
    informational text injected into a prompt, never executed, so an
    imperfect label costs nothing beyond being slightly less helpful.

    Capped at MAX_CONTEXT_SIGNATURES total and MAX_CONTEXT_SIBLING_FILES
    sibling files scanned — repo mode can have arbitrarily many sibling
    files, and this is meant to give the model a FEW grounding examples,
    not the whole codebase (which would blow out the prompt and defeat
    the purpose: a wall of unrelated signatures is noise, not grounding)."""
    signatures: list[str] = []
    seen: set[str] = set()

    def _collect(source: bytes, skip_range: tuple[int, int] | None) -> bool:
        for c in handler.chunk(source):
            if skip_range is not None and (c.start_byte, c.end_byte) == skip_range:
                continue  # the chunk being modernized itself
            first_line = c.code.splitlines()[0].strip() if c.code.strip() else ""
            if first_line and first_line not in seen:
                seen.add(first_line)
                signatures.append(first_line)
            if len(signatures) >= MAX_CONTEXT_SIGNATURES:
                return True
        return False

    if _collect(full_source, (chunk_start, chunk_end)):
        return signatures
    for sibling_source in (sibling_sources or [])[:MAX_CONTEXT_SIBLING_FILES]:
        if _collect(sibling_source, None):
            return signatures
    return signatures


def _extract_referenced_type_definitions(
    handler, chunk_code: str, full_source: bytes, sibling_sources: list[bytes] | None,
) -> list[str]:
    """Full source text of every class/struct/interface definition whose
    NAME appears as a whole word inside chunk_code (the function being
    modernized) — a deeper form of the same grounding
    _extract_context_signatures provides: instead of just the type's
    NAME, the model sees its actual fields/methods, so it can safely
    apply type hints or construct/use the type correctly instead of
    guessing at its shape. Scoped to the current file plus a bounded
    sibling sample (MAX_CONTEXT_SIBLING_FILES, same repo-mode-can-have-
    arbitrarily-many-files reasoning as _extract_context_signatures).

    Word-containment (not formal parameter-type-hint parsing) is a
    deliberate trade-off: precisely parsing "this chunk's declared
    parameter types" needs a DIFFERENT exact query per language grammar
    (Python type hints, TS type annotations, Java/C++/PHP typed
    parameters are all shaped differently) for a benefit this simpler
    check already delivers — a type referenced ANYWHERE in the chunk (a
    parameter type, a local variable's type, a `new Foo()` call, a
    static method reference) is exactly as useful to ground as one
    referenced only in the signature, and a whole-word regex match
    works identically across every language's syntax rather than
    needing 5 different implementations. Uses handler.
    type_definition_query_src (see languages/base.py) — empty for a
    language that doesn't define it, degrading to an empty result the
    same way every other "can I do this for this language" check in
    this project does."""
    if not handler.type_definition_query_src:
        return []

    types_found: dict[str, tuple[bytes, int, int]] = {}
    for name, (start, end) in handler.extract_type_definitions(full_source).items():
        types_found.setdefault(name, (full_source, start, end))
    for sibling_source in (sibling_sources or [])[:MAX_CONTEXT_SIBLING_FILES]:
        for name, (start, end) in handler.extract_type_definitions(sibling_source).items():
            types_found.setdefault(name, (sibling_source, start, end))

    definitions = []
    for name, (source, start, end) in types_found.items():
        if len(definitions) >= MAX_TYPE_DEFINITIONS:
            break
        if re.search(rf"\b{re.escape(name)}\b", chunk_code):
            definitions.append(source[start:end].decode("utf-8"))
    return definitions


def _punted_initial_state(
    language: str, full_source: bytes, chunk_start: int, chunk_end: int,
    original_code: str, max_iterations: int, recipe_instruction: str | None, punt_reason: str,
) -> AgentState:
    """A chunk skipped by assess_punt BEFORE any rewrite attempt — same
    terminal "gave_up" status every other unwritten chunk gets (main.py
    only branches on status/punted, not on a THIRD status value), just
    with iteration_count staying 0 and punted=True so callers can tell
    the difference. Every other field is a safe, unused default: nothing
    downstream reads probes/candidate_codes/etc. for a chunk this state
    machine never actually entered the graph for."""
    return {
        "language": language, "full_source": full_source,
        "chunk_start": chunk_start, "chunk_end": chunk_end,
        "original_code": original_code, "modernized_code": "", "required_imports": [],
        "candidate_codes": [], "baseline_stdout": None, "probes": [],
        "context_signatures": [], "referenced_type_definitions": [],
        "used_escalation": False, "used_deterministic_rule": False,
        "risk_flag": False, "risk_reason": "", "security_flag": False, "security_findings": [],
        "mutation_confidence_flag": False, "mutation_confidence_reason": "",
        "review_thread_id": None,
        "compiler_stderr": f"Punted before attempting (pre-rewrite confidence check): {punt_reason}",
        "iteration_count": 0, "status": "gave_up", "max_iterations": max_iterations,
        "recipe_instruction": recipe_instruction, "punted": True,
    }


def modernize(
    language: str,
    full_source: bytes,
    chunk_start: int,
    chunk_end: int,
    max_iterations: int = 5,
    sibling_sources: list[bytes] | None = None,
    interactive: bool = False,
    recipe_instruction: str | None = None,
    punt_check_enabled: bool = False,
) -> AgentState:
    original_code = full_source[chunk_start:chunk_end].decode("utf-8")

    if punt_check_enabled:
        # Runs BEFORE baseline/probe capture too (not just before the
        # graph) — those already cost sandbox time, and there's no point
        # spending it on a chunk about to be skipped anyway.
        punt_flag, punt_reason = assess_punt(language, original_code)
        if punt_flag:
            print(f"    (punted before attempting: {punt_reason})")
            return _punted_initial_state(
                language, full_source, chunk_start, chunk_end,
                original_code, max_iterations, recipe_instruction, punt_reason,
            )

    app = build_graph()
    handler = get_handler_by_name(language)
    baseline_stdout = _capture_baseline_stdout(language, full_source)
    probes = _capture_function_probes(language, full_source, original_code, sibling_sources)
    context_signatures = _extract_context_signatures(handler, full_source, chunk_start, chunk_end, sibling_sources)
    referenced_type_definitions = _extract_referenced_type_definitions(handler, original_code, full_source, sibling_sources)
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
        "context_signatures": context_signatures,
        "referenced_type_definitions": referenced_type_definitions,
        "used_escalation": False,
        "used_deterministic_rule": False,
        "punted": False,
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
        "recipe_instruction": recipe_instruction,
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
