import os
import re
import time
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from sandbox.verifier import verify, run_semgrep
from languages import get_handler_by_name

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

# Optional escalation to a stronger model after the cheap one keeps
# failing on the SAME chunk. Off by default — behavior is completely
# unchanged unless ESCALATION_MODEL is explicitly set. This targets the
# actual bottleneck we've observed all session: the pipeline mechanics
# are sound, the 7B model's judgment is the ceiling on quality. Escalating
# only after N proven failures means you pay for the stronger model only
# where the cheap one already showed it can't do the job.
ESCALATION_MODEL = os.environ.get("ESCALATION_MODEL")
ESCALATION_THRESHOLD = int(os.environ.get("ESCALATION_THRESHOLD", "3"))
escalation_llm = ChatOllama(model=ESCALATION_MODEL, temperature=0) if ESCALATION_MODEL else None

# Execution-grounded best-of-N: on the FIRST (blind, no error feedback)
# attempt only, generate this many independent candidates and let
# verifier_node run the real pipeline (structural + sandbox + behavioral
# + probes + determinism) against each, keeping whichever one actually
# passes instead of committing to a single guess. The extra candidate(s)
# come from _diversity_llm — a separate, higher-temperature instance of
# the SAME base model — because sampling the deterministic (temperature=0)
# `llm` twice with an identical prompt would return the same response
# both times, defeating the purpose.
BEST_OF_N_ON_FIRST_ATTEMPT = int(os.environ.get("BEST_OF_N_ON_FIRST_ATTEMPT", "2"))
_diversity_llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.7)

_LLM_RETRY_ATTEMPTS = 3
_LLM_RETRY_DELAY_SECONDS = 3


def _select_llm(iteration_count: int):
    if escalation_llm is not None and iteration_count >= ESCALATION_THRESHOLD:
        return escalation_llm, True
    return llm, False


def _invoke_llm_with_retry(llm_instance, messages):
    """The Ollama server is a local background process we don't control —
    we've seen it die mid-run (e.g. the machine slept, or it crashed) and
    take the whole main.py run down with a raw ConnectionError. Retry a
    few times with a short delay before giving up, so one transient blip
    doesn't waste an otherwise-successful multi-chunk run."""
    last_error = None
    for attempt in range(_LLM_RETRY_ATTEMPTS):
        try:
            return llm_instance.invoke(messages)
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < _LLM_RETRY_ATTEMPTS - 1:
                time.sleep(_LLM_RETRY_DELAY_SECONDS)
    raise ConnectionError(
        f"Could not reach Ollama after {_LLM_RETRY_ATTEMPTS} attempts — "
        f"is `ollama serve` running? Original error: {last_error}"
    ) from last_error

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$", re.MULTILINE)
# Matches either comment style so one parser works for every language:
# "// REQUIRES: module" (C++/Java/JS/TS/PHP) or "# REQUIRES: module" (Python)
_REQUIRES_RE = re.compile(r"^\s*(?:#|//)\s*REQUIRES:\s*(\S+)\s*$", re.MULTILINE)


def _strip_markdown_fence(text: str) -> str:
    """Local models are less reliable than Claude at following
    'no markdown fences' instructions, so strip them defensively."""
    return _FENCE_RE.sub("", text.strip()).strip()


def _extract_required_imports(text: str) -> tuple[str, list[str]]:
    """Pull out `REQUIRES: module` marker lines the model emits when it
    needs a new import/header. Returns (code_with_markers_removed,
    [module, ...]). This lets the model request dependencies without
    letting it place raw import statements inline in the function body,
    which would corrupt the splice."""
    modules = _REQUIRES_RE.findall(text)
    clean = _REQUIRES_RE.sub("", text).strip()
    return clean, modules


def refactorer_node(state: AgentState) -> AgentState:
    handler = get_handler_by_name(state["language"])

    if state["iteration_count"] == 0:
        messages = [
            SystemMessage(content=handler.refactor_system_prompt),
            HumanMessage(content=state["original_code"]),
        ]
    else:
        messages = [
            SystemMessage(content=handler.fix_system_prompt),
            HumanMessage(
                content=(
                    f"Previous attempt:\n{state['modernized_code']}\n\n"
                    f"Compiler/runtime error:\n{state['compiler_stderr']}\n\n"
                    f"Original legacy code (for reference):\n{state['original_code']}"
                )
            ),
        ]

    selected_llm, used_escalation = _select_llm(state["iteration_count"])
    if used_escalation:
        print(f"    (escalating to {ESCALATION_MODEL} after {state['iteration_count']} failed attempts)")

    llms_to_try = [selected_llm]
    if state["iteration_count"] == 0 and not used_escalation:
        # Best-of-N ONLY on the first, blind attempt. Once there's real
        # compiler/behavioral error feedback (any retry) or we've already
        # escalated to a stronger model, a second independent blind guess
        # adds sandbox cost without the same diversity payoff — at that
        # point the model has specific information to act on, which
        # matters more than another untargeted guess. The extra
        # candidate(s) use a HIGHER-temperature instance of the same
        # model: at temperature=0 (the default `llm`), an identical
        # prompt returns an all-but-identical response, so reusing
        # selected_llm again here would mostly just waste an LLM call.
        llms_to_try.extend([_diversity_llm] * (BEST_OF_N_ON_FIRST_ATTEMPT - 1))

    candidates = []
    for candidate_llm in llms_to_try:
        response = _invoke_llm_with_retry(candidate_llm, messages)
        fenced = _strip_markdown_fence(response.content)
        clean_code, required_imports = _extract_required_imports(fenced)
        candidates.append({"code": clean_code, "required_imports": required_imports})

    return {
        **state,
        "candidate_codes": candidates,
        # Kept populated with candidate 0 as a reasonable default for any
        # code that reads modernized_code directly — verifier_node
        # authoritatively overwrites both fields with whichever candidate
        # actually wins once verification runs.
        "modernized_code": candidates[0]["code"],
        "required_imports": candidates[0]["required_imports"],
        "used_escalation": used_escalation or state.get("used_escalation", False),
    }


def _validate_single_chunk(handler, code: str) -> str | None:
    """Re-parse the model's output on its own and confirm it's exactly
    ONE function/method spanning (essentially) the whole response — catches
    the model appending stray extra functions, duplicated code, or literal
    import statements it was told not to write. A whole-file compile/run
    check alone won't catch this: many languages (Python included) happily
    tolerate duplicate top-level defs at runtime, so a corrupted response
    can still report "success". Returns an error string, or None if valid."""
    # Some grammars (PHP) refuse to recognize bare code without a wrapper
    # tag, so prepend it only for this parse — never part of the actual
    # splice, which uses `code` (unwrapped) as-is.
    prefix_bytes = handler.parse_wrapper_prefix.encode("utf-8")
    code_bytes = code.encode("utf-8")
    wrapped_bytes = prefix_bytes + code_bytes

    sub_chunks = handler.chunk(wrapped_bytes)
    if len(sub_chunks) != 1:
        return (
            f"Your response must contain exactly ONE function/method, but "
            f"it parsed as {len(sub_chunks)}. Output only the single "
            f"rewritten function — no extra functions, no duplicated code, "
            f"no literal import statements (use a REQUIRES marker instead)."
        )
    c = sub_chunks[0]
    # Nothing may exist outside the matched function's byte range (besides
    # the wrapper prefix itself) — not even a short stray import line,
    # which a length-ratio check could miss.
    before = wrapped_bytes[len(prefix_bytes):c.start_byte].strip()
    after = wrapped_bytes[c.end_byte:].strip()
    if before or after:
        return (
            "Your response contains extra content outside the function "
            "body (e.g. a literal import statement, stray text, or another "
            "declaration). Output only the single rewritten function, "
            "nothing else — request imports via a REQUIRES marker instead."
        )
    return None


def _verify_candidate(handler, state: AgentState, code: str, required_imports: list[str]) -> dict:
    """Run the full verification pipeline (structural validation, sandbox
    compile/run, whole-file behavioral equivalence, per-probe checks,
    determinism re-check) against ONE candidate. Returns
    {"status": "success"|"failed", "compiler_stderr": str}. Extracted out
    of what used to be verifier_node's entire body so it can be called
    once per candidate under best-of-N (see refactorer_node's
    BEST_OF_N_ON_FIRST_ATTEMPT) without duplicating five layers of
    checking logic for each one."""
    validation_error = _validate_single_chunk(handler, code)
    if validation_error is not None:
        return {"status": "failed", "compiler_stderr": validation_error}

    # Splice the candidate chunk back into the FULL original file and
    # compile/run that — a chunk (one function/method) is not an
    # independently runnable unit on its own.
    candidate_file = handler.build_candidate(
        full_source=state["full_source"],
        chunk_start=state["chunk_start"],
        chunk_end=state["chunk_end"],
        modernized_code=code,
        required_imports=required_imports,
    )

    result = verify(
        candidate_file.decode("utf-8"),
        filename=handler.sandbox_filename,
        run_cmd=handler.run_command(),
    )

    baseline_stdout = state.get("baseline_stdout")
    if result["status"] == "success" and baseline_stdout is not None:
        if result["stdout"] != baseline_stdout:
            return {
                "status": "failed",
                "compiler_stderr": (
                    "Your rewritten function compiles and runs, but changes "
                    "the program's output — modernization must not change "
                    "observable behavior.\n\n"
                    f"Expected stdout:\n{baseline_stdout!r}\n\n"
                    f"Got:\n{result['stdout']!r}\n\n"
                    "Fix the function so it produces IDENTICAL output to "
                    "the original."
                ),
            }

    # Whole-file baseline only proves equivalence for whatever the file's
    # entry point exercises. Each probe (real call site or LLM-synthesized
    # example — see _capture_function_probes) is run against the
    # MODERNIZED candidate and compared to its own baseline. Checking
    # every probe, not just one, is the direct fix for "a single example
    # proves nothing about edge cases."
    probes = state.get("probes") or []
    if result["status"] == "success" and probes:
        for i, probe in enumerate(probes):
            snippet = probe["snippet"]
            baseline = probe["baseline_stdout"]
            probe_candidate = candidate_file + b"\n" + snippet.encode("utf-8") + b"\n"
            probe_result = verify(
                probe_candidate.decode("utf-8"),
                filename=handler.sandbox_filename,
                run_cmd=handler.run_command(),
            )
            if probe_result["status"] != "success" or probe_result["stdout"] != baseline:
                return {
                    "status": "failed",
                    "compiler_stderr": (
                        "Your rewritten function compiles and the whole file "
                        "still runs, but calling it directly with representative "
                        "arguments produces a different result than the original "
                        "function did.\n\n"
                        f"Probe used: {snippet}\n\n"
                        f"Expected: {baseline!r}\n\n"
                        f"Got: {probe_result.get('stdout', probe_result.get('stderr'))!r}\n\n"
                        "Fix the function so it produces IDENTICAL results for "
                        "the same inputs."
                    ),
                }

            if i == 0:
                # Determinism re-check: run the SAME probe against the SAME
                # modernized candidate a SECOND time and confirm it matches
                # ITSELF, not just the baseline. Only done once per chunk
                # (not once per probe) to avoid multiplying sandbox calls —
                # this catches non-determinism the modernization may have
                # introduced (dict-ordering dependence, uninitialized
                # state, a subtle race) that happened to match the
                # baseline on this one run without actually being stable.
                probe_result_2 = verify(
                    probe_candidate.decode("utf-8"),
                    filename=handler.sandbox_filename,
                    run_cmd=handler.run_command(),
                )
                if probe_result_2["status"] != "success" or probe_result_2["stdout"] != probe_result["stdout"]:
                    return {
                        "status": "failed",
                        "compiler_stderr": (
                            "Your rewritten function produces DIFFERENT output "
                            "across two identical runs with the same input — "
                            "this indicates non-deterministic behavior (e.g. "
                            "dependence on dict/set ordering, uninitialized "
                            "state, or a race condition) that the modernization "
                            "introduced.\n\n"
                            f"Probe used: {snippet}\n\n"
                            f"Run 1: {probe_result['stdout']!r}\n\n"
                            f"Run 2: {probe_result_2.get('stdout', probe_result_2.get('stderr'))!r}\n\n"
                            "Fix the function so it produces the SAME output "
                            "every time for the same input."
                        ),
                    }

    return {"status": result["status"], "compiler_stderr": result["stderr"]}


def verifier_node(state: AgentState) -> AgentState:
    handler = get_handler_by_name(state["language"])

    candidates = state.get("candidate_codes") or [
        {"code": state["modernized_code"], "required_imports": state.get("required_imports", [])}
    ]

    results = []
    for candidate in candidates:
        result = _verify_candidate(handler, state, candidate["code"], candidate["required_imports"])
        results.append(result)
        if result["status"] == "success":
            if len(candidates) > 1:
                print(f"    (best-of-{len(candidates)}: candidate {len(results)} passed)")
            return {
                **state,
                "modernized_code": candidate["code"],
                "required_imports": candidate["required_imports"],
                "compiler_stderr": result["compiler_stderr"],
                "status": "success",
                "iteration_count": state["iteration_count"] + 1,
            }

    # None of the candidates passed. Report the FIRST (deterministic,
    # temperature=0) candidate's error for the fix-prompt on retry — an
    # arbitrary but consistent choice among N failures, rather than e.g.
    # whichever failed "most interestingly."
    first_candidate, first_result = candidates[0], results[0]
    return {
        **state,
        "modernized_code": first_candidate["code"],
        "required_imports": first_candidate["required_imports"],
        "compiler_stderr": first_result["compiler_stderr"],
        "status": "failed",
        "iteration_count": state["iteration_count"] + 1,
    }


_RISK_SYSTEM_PROMPT = """You just modernized a function. Assess whether the
sandbox's behavioral check (comparing the file's stdout before and after)
would actually catch a regression in THIS function if you introduced one.

Answer with EXACTLY two lines:
RISK: yes
<one short sentence why>

or:

RISK: no
<one short sentence why>

Answer RISK: yes if the function touches ANY of: file I/O, network calls,
database access, global/static mutable state, randomness, the system clock
or dates, environment variables, or concurrency — categories where two runs
producing the same stdout does NOT prove the behavior is actually
equivalent (e.g. a function with a subtle race condition or an off-by-one
date bug can still print identical output on a single test run).

Answer RISK: no only if the function is pure — its output depends only on
its inputs, with no side effects and nothing that could vary between runs."""

_RISK_RE = re.compile(r"RISK:\s*(yes|no)", re.IGNORECASE)


def assess_risk(modernized_code: str) -> tuple[bool, str]:
    """One extra LLM call, made only after a chunk has already passed
    structural validation, sandbox compile/run, AND the stdout-equivalence
    check. Those checks prove "this specific run behaved the same" — they
    can't prove "this function can never behave differently," which matters
    for anything with side effects or non-determinism. Always uses the
    base model regardless of escalation state: this is a cheap self-critique
    prompt, not a task that benefits from a stronger model, and running it
    on the escalation model would just double the cost of already-escalated
    chunks for no real benefit."""
    messages = [
        SystemMessage(content=_RISK_SYSTEM_PROMPT),
        HumanMessage(content=modernized_code),
    ]
    response = _invoke_llm_with_retry(llm, messages)
    text = response.content.strip()
    match = _RISK_RE.search(text)
    risk_flag = bool(match) and match.group(1).lower() == "yes"
    return risk_flag, text


def scan_security(language: str, modernized_code: str) -> tuple[bool, list[dict]]:
    """Static-analysis security scan (semgrep, local offline rules — see
    sandbox/security-rules.yaml), run only after a chunk has already
    passed every behavioral check. Behavioral equivalence (whole-file
    diff, probes, determinism) proves "does the same thing" — none of
    it proves "doesn't introduce a NEW vulnerability while doing the
    same thing." A modernization can pass every existing check while
    swapping a safe pattern for an unsafe one (e.g. parameterized
    formatting into a raw f-string used in a shell command), which is
    exactly the gap this closes. Flags, doesn't block: semgrep's
    precision isn't perfect (industry benchmarks put community rulesets
    around 60-65% correct-match rate), so a false positive here must
    not be able to reject an otherwise-correct modernization — this
    mirrors assess_risk's flag-not-block design, not the hard-fail
    design used for compile/behavioral checks.

    Scans under the target language's own sandbox_filename (e.g.
    "main.php", not always "main.py") — semgrep selects which of the
    12 rules to apply based on the file EXTENSION, so scanning PHP code
    under a .py filename would silently apply only the Python rules and
    find nothing, regardless of what the code actually contains."""
    handler = get_handler_by_name(language)
    result = run_semgrep(modernized_code, handler.sandbox_filename)
    if result["status"] != "success":
        return False, []
    return bool(result["findings"]), result["findings"]


def fallback_node(state: AgentState) -> AgentState:
    return {**state, "status": "gave_up"}


_PROBE_LANGUAGE_LABELS = {
    "python": ("Python", "print(...)"),
    "javascript": ("JavaScript", "console.log(...)"),
    "typescript": ("TypeScript", "console.log(...)"),
    "php": ("PHP", "echo ...;"),
}

_PROBE_SYSTEM_PROMPT_TEMPLATE = """You will be given ONE {label} function.
Write {count} short snippets, each on its own line, that call this
function with DIFFERENT example arguments and print/output the result —
so its behavior can be compared across a RANGE of inputs, not just one.
A single example proves almost nothing about edge cases; {count} varied
ones is a real check.

Rules:
- Each line is a complete, independent call+print statement — nothing
  else on that line.
- Call the function using its exact name as shown.
- Make the {count} lines meaningfully different: cover a typical case,
  an edge case (empty string / zero / negative / boundary value), and
  another distinct typical case — don't just change one digit each time.
- Print/output the result using this language's normal mechanism: {print_call}
- Do NOT redefine the function. Do NOT add imports or requires. Do NOT add
  explanatory text, numbering, or markdown fences.
- If the function needs objects/types too complex to construct from
  scratch (e.g. it takes a database connection), respond with EXACTLY:
  PROBE: SKIP

Respond with ONLY the {count} lines (or PROBE: SKIP), nothing else."""


def generate_probes(language: str, function_code: str, count: int = 3) -> list[str]:
    """Ask the model for MULTIPLE diverse 'call this function, print the
    result' snippets instead of just one. A single hand-picked example
    proves almost nothing about edge cases (empty/zero/negative/boundary
    values) it never exercises — this directly targets that gap, in the
    same spirit as differential-fuzzing approaches to verifying LLM code
    refactorings (generate many inputs, compare outputs across all of
    them, not just one). Returns [] if the model can't/won't produce
    any; callers must treat that as "no synthesized probes available,"
    not an error."""
    label, print_call = _PROBE_LANGUAGE_LABELS[language]
    prompt = _PROBE_SYSTEM_PROMPT_TEMPLATE.format(label=label, print_call=print_call, count=count)
    messages = [SystemMessage(content=prompt), HumanMessage(content=function_code)]
    response = _invoke_llm_with_retry(llm, messages)
    text = _strip_markdown_fence(response.content).strip()
    if not text or "SKIP" in text.upper():
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:count]


_PROBE_WRAPPERS = {
    "python": lambda call: f"print({call})",
    "javascript": lambda call: f"console.log({call})",
    "typescript": lambda call: f"console.log({call})",
    "php": lambda call: f"echo {call};",
}


def wrap_call_as_probe(language: str, call_expression: str) -> str:
    """Turn a bare call expression (e.g. from a real call site found in
    the codebase, which is just the expression text with no print/echo
    around it) into a runnable probe statement."""
    return _PROBE_WRAPPERS[language](call_expression)
