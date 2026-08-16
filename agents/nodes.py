import os
import re
import time
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from sandbox.verifier import verify
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

    response = _invoke_llm_with_retry(selected_llm, messages)
    fenced = _strip_markdown_fence(response.content)
    clean_code, required_imports = _extract_required_imports(fenced)
    return {
        **state,
        "modernized_code": clean_code,
        "required_imports": required_imports,
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


def verifier_node(state: AgentState) -> AgentState:
    handler = get_handler_by_name(state["language"])

    validation_error = _validate_single_chunk(handler, state["modernized_code"])
    if validation_error is not None:
        return {
            **state,
            "compiler_stderr": validation_error,
            "status": "failed",
            "iteration_count": state["iteration_count"] + 1,
        }

    # Splice the candidate chunk back into the FULL original file and
    # compile/run that — a chunk (one function/method) is not an
    # independently runnable unit on its own.
    candidate_file = handler.build_candidate(
        full_source=state["full_source"],
        chunk_start=state["chunk_start"],
        chunk_end=state["chunk_end"],
        modernized_code=state["modernized_code"],
        required_imports=state.get("required_imports", []),
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
                **state,
                "compiler_stderr": (
                    "Your rewritten function compiles and runs, but changes "
                    "the program's output — modernization must not change "
                    "observable behavior.\n\n"
                    f"Expected stdout:\n{baseline_stdout!r}\n\n"
                    f"Got:\n{result['stdout']!r}\n\n"
                    "Fix the function so it produces IDENTICAL output to "
                    "the original."
                ),
                "status": "failed",
                "iteration_count": state["iteration_count"] + 1,
            }

    # Whole-file baseline only proves equivalence for whatever the file's
    # entry point exercises. If a probe is available (this chunk's
    # function was reachable enough to generate one for), run it against
    # the MODERNIZED candidate and compare — this catches regressions in
    # functions the file's own runtime path never touches at all.
    probe_snippet = state.get("probe_snippet")
    probe_baseline_stdout = state.get("probe_baseline_stdout")
    if result["status"] == "success" and probe_snippet and probe_baseline_stdout is not None:
        probe_candidate = candidate_file + b"\n" + probe_snippet.encode("utf-8") + b"\n"
        probe_result = verify(
            probe_candidate.decode("utf-8"),
            filename=handler.sandbox_filename,
            run_cmd=handler.run_command(),
        )
        probe_stdout_matches = (
            probe_result["status"] == "success" and probe_result["stdout"] == probe_baseline_stdout
        )
        if not probe_stdout_matches:
            return {
                **state,
                "compiler_stderr": (
                    "Your rewritten function compiles and the whole file "
                    "still runs, but calling it directly with representative "
                    "arguments produces a different result than the original "
                    "function did.\n\n"
                    f"Probe used: {probe_snippet}\n\n"
                    f"Expected: {probe_baseline_stdout!r}\n\n"
                    f"Got: {probe_result.get('stdout', probe_result.get('stderr'))!r}\n\n"
                    "Fix the function so it produces IDENTICAL results for "
                    "the same inputs."
                ),
                "status": "failed",
                "iteration_count": state["iteration_count"] + 1,
            }

    return {
        **state,
        "compiler_stderr": result["stderr"],
        "status": result["status"],
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


def fallback_node(state: AgentState) -> AgentState:
    return {**state, "status": "gave_up"}


_PROBE_LANGUAGE_LABELS = {
    "python": ("Python", "print(...)"),
    "javascript": ("JavaScript", "console.log(...)"),
    "typescript": ("TypeScript", "console.log(...)"),
    "php": ("PHP", "echo ...;"),
}

_PROBE_SYSTEM_PROMPT_TEMPLATE = """You will be given ONE {label} function.
Write exactly one short snippet that calls this function with realistic,
concrete example arguments and prints/outputs the result, so its behavior
can be observed and compared before vs. after a future change.

Rules:
- Call the function using its exact name as shown.
- Choose simple, concrete argument values (numbers, short strings) — not
  placeholders, not randomly generated values.
- Print/output the result using this language's normal mechanism: {print_call}
- Do NOT redefine the function. Do NOT add imports or requires. Do NOT add
  explanatory text or markdown fences.
- If the function needs objects/types too complex to construct from
  scratch (e.g. it takes a database connection), respond with EXACTLY:
  PROBE: SKIP

Respond with ONLY the call+print snippet (or PROBE: SKIP), nothing else."""


def generate_probe(language: str, function_code: str) -> str | None:
    """Ask the model to write one small 'call this function, print the
    result' snippet. This exists to close a real gap: the whole-file
    behavioral check only proves equivalence for whatever the file's own
    entry point happens to exercise — a function nothing in the file
    currently calls gets zero coverage from that check. Returns None if
    the model can't/won't produce one; callers must treat that as "no
    probe available," not an error — this is a best-effort addition on
    top of checks that already provide the real safety net."""
    label, print_call = _PROBE_LANGUAGE_LABELS[language]
    prompt = _PROBE_SYSTEM_PROMPT_TEMPLATE.format(label=label, print_call=print_call)
    messages = [SystemMessage(content=prompt), HumanMessage(content=function_code)]
    response = _invoke_llm_with_retry(llm, messages)
    text = _strip_markdown_fence(response.content).strip()
    if not text or "SKIP" in text.upper():
        return None
    return text
