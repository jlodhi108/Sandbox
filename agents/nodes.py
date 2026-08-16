import os
import re
import time
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from sandbox.verifier import verify, run_semgrep
from languages import get_handler_by_name
from llm_budget import LLMBudget
import deterministic_rules
import property_testing

# Default ceiling for THIS process's lifetime, from MAX_LLM_CALLS_PER_RUN
# if set (parity with every other env-var-configurable setting in this
# module). main.py's __main__ block and mcp_server.py's tool functions
# both call llm_budget.reset(max_calls=...) with the config-resolved
# value at the start of each actual run, so this import-time default
# only matters for callers that skip that step (e.g. calling agents.graph
# directly, as the test suite does).
MAX_LLM_CALLS_PER_RUN = os.environ.get("MAX_LLM_CALLS_PER_RUN")
llm_budget = LLMBudget(max_calls=int(MAX_LLM_CALLS_PER_RUN) if MAX_LLM_CALLS_PER_RUN else None)

# Ollama inference params, sized for qwen2.5-coder:14b rather than the
# library's defaults (2048 ctx / unbounded predict — too small a context
# window truncates larger chunks silently instead of erroring, which is
# worse than being explicit here). Both env-overridable for parity with
# every other tunable in this module.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))

llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
)

# Optional escalation to a stronger model after the cheap one keeps
# failing on the SAME chunk. Off by default — behavior is completely
# unchanged unless ESCALATION_MODEL is explicitly set. This targets the
# actual bottleneck we've observed all session: the pipeline mechanics
# are sound, the 7B model's judgment is the ceiling on quality. Escalating
# only after N proven failures means you pay for the stronger model only
# where the cheap one already showed it can't do the job.
ESCALATION_MODEL = os.environ.get("ESCALATION_MODEL")
ESCALATION_THRESHOLD = int(os.environ.get("ESCALATION_THRESHOLD", "3"))
escalation_llm = (
    ChatOllama(model=ESCALATION_MODEL, temperature=0, num_ctx=OLLAMA_NUM_CTX, num_predict=OLLAMA_NUM_PREDICT)
    if ESCALATION_MODEL else None
)

# Optional dedicated model for assess_risk()'s self-critique step. Research
# on adversarial/verifier-pattern code review is explicit that a model
# reviewing its own output is a weak checker — it measurably favors its
# own generative patterns regardless of how capable it is, which is a
# different axis than "is this model smart enough" (what ESCALATION_MODEL
# targets). See _select_reviewer_llm for how this combines with
# escalation_llm to get a genuinely different model for free when one is
# already configured, with zero new setup required.
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL")
reviewer_llm = (
    ChatOllama(model=REVIEWER_MODEL, temperature=0, num_ctx=OLLAMA_NUM_CTX, num_predict=OLLAMA_NUM_PREDICT)
    if REVIEWER_MODEL else None
)

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
_diversity_llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0.7,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
)

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
    doesn't waste an otherwise-successful multi-chunk run.

    This is the single chokepoint every LLM call in this codebase passes
    through (refactor, fix, risk assessment, probe generation) — the
    right, and only, place to enforce a call budget and count usage,
    rather than instrumenting every call site separately. check() raises
    BEFORE making a new call if the run is already at/over budget;
    record() runs after a successful call, using response.usage_metadata
    (confirmed present on ChatOllama responses: input_tokens/output_tokens/
    total_tokens) when the backend provides it."""
    llm_budget.check()
    last_error = None
    for attempt in range(_LLM_RETRY_ATTEMPTS):
        try:
            response = llm_instance.invoke(messages)
            llm_budget.record(getattr(llm_instance, "model", "unknown"), getattr(response, "usage_metadata", None))
            return response
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


# --- REQUIRES-marker resolvability check ---------------------------------
#
# The sandbox that runs candidate code has NO network access and never
# runs an install step for a per-run dependency (confirmed by testing
# directly against the sandbox image: `npm install` fails outright with
# no registry reachable, and nothing in the verify() pipeline ever runs
# `pip install`/`npm install` for a candidate run). That means a REQUIRES
# marker can only ever succeed here if it names a module the language
# runtime ships with built in. Whether a third-party name is a REAL,
# registered PyPI/npm package or a fully hallucinated one is beside the
# point — either way it can never resolve in THIS sandbox, so checking it
# against the public registry would be actively misleading: it would
# report a perfectly real package as "fine" moments before the sandbox
# fails it with ModuleNotFoundError/MODULE_NOT_FOUND anyway. So this
# checks against each runtime's actual builtin module set instead — a
# fast, exact, zero-network lookup that answers the question that
# actually matters here ("can this resolve IN THIS SANDBOX"), not "does
# this exist somewhere in the world".
#
# Snapshotted directly from the sandbox-multi image (not hand-typed):
#   python3 -c "import sys; sorted(m for m in sys.stdlib_module_names if not m.startswith('_'))"
#   node -e "require('node:module').builtinModules.filter(m => !m.startsWith('_'))"
# Regenerate both if sandbox/sandbox-multi.Dockerfile's Python or Node
# version ever changes.
_PYTHON_STDLIB_MODULES = frozenset({
    'abc', 'aifc', 'antigravity', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
    'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'bisect', 'builtins',
    'bz2', 'cProfile', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd', 'code',
    'codecs', 'codeop', 'collections', 'colorsys', 'compileall', 'concurrent',
    'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg', 'crypt', 'csv',
    'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal', 'difflib', 'dis',
    'distutils', 'doctest', 'email', 'encodings', 'ensurepip', 'enum', 'errno',
    'faulthandler', 'fcntl', 'filecmp', 'fileinput', 'fnmatch', 'fractions', 'ftplib',
    'functools', 'gc', 'genericpath', 'getopt', 'getpass', 'gettext', 'glob', 'graphlib',
    'grp', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib',
    'imghdr', 'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
    'keyword', 'lib2to3', 'linecache', 'locale', 'logging', 'lzma', 'mailbox', 'mailcap',
    'marshal', 'math', 'mimetypes', 'mmap', 'modulefinder', 'msilib', 'msvcrt',
    'multiprocessing', 'netrc', 'nis', 'nntplib', 'nt', 'ntpath', 'nturl2path', 'numbers',
    'opcode', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib', 'pdb', 'pickle',
    'pickletools', 'pipes', 'pkgutil', 'platform', 'plistlib', 'poplib', 'posix',
    'posixpath', 'pprint', 'profile', 'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr',
    'pydoc', 'pydoc_data', 'pyexpat', 'queue', 'quopri', 'random', 're', 'readline',
    'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched', 'secrets', 'select',
    'selectors', 'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
    'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'sre_compile', 'sre_constants',
    'sre_parse', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny', 'tarfile',
    'telnetlib', 'tempfile', 'termios', 'textwrap', 'this', 'threading', 'time', 'timeit',
    'tkinter', 'token', 'tokenize', 'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty',
    'turtle', 'turtledemo', 'types', 'typing', 'unicodedata', 'unittest', 'urllib', 'uu',
    'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser', 'winreg', 'winsound',
    'wsgiref', 'xdrlib', 'xml', 'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib',
    'zoneinfo',
})

_NODE_BUILTIN_MODULES = frozenset({
    'assert', 'assert/strict', 'async_hooks', 'buffer', 'child_process', 'cluster',
    'console', 'constants', 'crypto', 'dgram', 'diagnostics_channel', 'dns',
    'dns/promises', 'domain', 'events', 'fs', 'fs/promises', 'http', 'http2', 'https',
    'inspector', 'inspector/promises', 'module', 'net', 'os', 'path', 'path/posix',
    'path/win32', 'perf_hooks', 'process', 'punycode', 'querystring', 'readline',
    'readline/promises', 'repl', 'stream', 'stream/consumers', 'stream/promises',
    'stream/web', 'string_decoder', 'sys', 'timers', 'timers/promises', 'tls',
    'trace_events', 'tty', 'url', 'util', 'util/types', 'v8', 'vm', 'wasi',
    'worker_threads', 'zlib',
})

# C++ headers, Java imports, and PHP namespaces have no equivalent "only
# the runtime's own modules are reachable" ambiguity worth checking here:
# a bad name in any of those three fails immediately and clearly at the
# existing compile step (verify(), right after this check runs) with a
# precise compiler error the fix-retry loop already handles well — so
# this only covers the languages where a wrong-but-plausible-looking
# name would otherwise silently burn a full sandbox round-trip first.
_REQUIRES_CHECK_LANGUAGES = {"python", "javascript", "typescript"}


def check_requires_resolvable(language: str, module: str) -> bool | None:
    """Can this REQUIRES: module marker possibly resolve in the sandbox?
    Returns True (it's a builtin — will resolve), False (it names a
    third-party package — can NEVER resolve here, regardless of whether
    it's a real registered package or a hallucinated one), or None
    (language not covered by this check — see _REQUIRES_CHECK_LANGUAGES,
    callers should treat None as "no opinion, let it through")."""
    if language not in _REQUIRES_CHECK_LANGUAGES:
        return None
    if language == "python":
        # A REQUIRES value may be a dotted submodule path (e.g.
        # "urllib.request", "concurrent.futures") — only the top-level
        # package name determines resolvability.
        top_level = module.split(".", 1)[0]
        return top_level in _PYTHON_STDLIB_MODULES
    # javascript / typescript: accept both the bare form ("fs") and the
    # explicit "node:" prefix ("node:fs") modern code increasingly uses.
    return module.removeprefix("node:") in _NODE_BUILTIN_MODULES


def _with_recipe(system_prompt: str, recipe_instruction: str | None) -> str:
    """Append a recipe's extra guidance to a language handler's base
    system prompt, if one is configured for this run (see
    AgentState.recipe_instruction and config.load_recipes). Appended
    rather than replacing the base prompt — the recipe narrows/steers
    WHAT kind of modernization happens (e.g. "only convert callbacks to
    async/await"), it doesn't need to restate the base rules every
    language handler already enforces (single-function output, REQUIRES
    markers, no markdown fences, etc.)."""
    if not recipe_instruction:
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"--- Additional guidance for this modernization run ---\n"
        f"{recipe_instruction}"
    )


def _format_context_block(context_signatures: list[str] | None) -> str:
    """Formats the sibling/codebase signature context (see
    agents/graph.py:_extract_context_signatures) into a prompt section,
    or '' if there's none to show. Grounds the model in what the
    codebase ACTUALLY has — real function/type names it can reference
    with confidence — instead of guessing, the same hallucination gap
    static-analysis-first approaches (AlphaTrans, LegacyTranslate) exist
    to close, scoped down to what a same-language, function-level
    pipeline actually needs: a short list of real signatures already
    extracted via tree-sitter chunking, not a full cross-language
    project skeleton (this project modernizes Python-to-Python,
    JS-to-JS, etc. — there's no cross-language type-resolution problem
    to solve here)."""
    if not context_signatures:
        return ""
    listed = "\n".join(f"- {s}" for s in context_signatures)
    return (
        "For reference, other functions/types already defined elsewhere in this "
        "codebase (do not redefine them; only call one if you're confident about "
        "its exact signature and behavior):\n"
        f"{listed}\n\n"
    )


def _format_type_definitions_block(referenced_type_definitions: list[str] | None) -> str:
    """Formats the FULL source of every class/struct/interface this
    chunk references (see agents/graph.py:
    _extract_referenced_type_definitions) into a prompt section, or ''
    if there's none. Deeper grounding than _format_context_block's
    one-line names: the model sees the type's actual fields/methods, so
    it can apply type hints or construct/use the type correctly instead
    of guessing at its shape."""
    if not referenced_type_definitions:
        return ""
    joined = "\n\n".join(referenced_type_definitions)
    return (
        "Full definitions of type(s) this function references, so you know their "
        "exact shape (do not redefine them):\n"
        f"{joined}\n\n"
    )


def _format_exemplar_block(exemplar_original: str | None, exemplar_modernized: str | None) -> str:
    """Formats the single most semantically-similar PAST successful
    modernization (see exemplar_bank.py) as a worked example, or '' if
    none is available. In-context learning from this project's own
    proven track record — shown ONLY on the first (blind) attempt, not
    retries: once there's real compiler/behavioral error feedback, that
    feedback is more directly useful than a generic worked example."""
    if not exemplar_original or not exemplar_modernized:
        return ""
    return (
        "Here is an example of a similar function this project has ALREADY "
        "successfully modernized (verified equivalent, from a past run) — use it "
        "as a style guide for the kind of transformation to make, not as literal "
        "code to copy:\n"
        f"Before:\n{exemplar_original}\n\n"
        f"After:\n{exemplar_modernized}\n\n"
    )


def refactorer_node(state: AgentState) -> AgentState:
    handler = get_handler_by_name(state["language"])
    recipe_instruction = state.get("recipe_instruction")
    context_block = (
        _format_type_definitions_block(state.get("referenced_type_definitions"))
        + _format_context_block(state.get("context_signatures"))
    )

    if state["iteration_count"] == 0:
        # Try a deterministic (non-LLM) rewrite FIRST, before spending any
        # LLM call at all — see deterministic_rules.py's module docstring
        # for exactly what makes a rule eligible (provably behavior-
        # preserving by construction, not a heuristic). This candidate
        # still goes through the EXACT SAME verification pipeline as an
        # LLM-generated one (verifier_node doesn't know or care where a
        # candidate came from) — this only ever skips the LLM call, never
        # a safety check. If it fails verification, the next iteration
        # falls through to the normal LLM path below with real error
        # feedback, same as any other failed attempt.
        deterministic_code = deterministic_rules.try_apply(state["language"], state["original_code"])
        if deterministic_code is not None:
            return {
                **state,
                "candidate_codes": [{"code": deterministic_code, "required_imports": []}],
                "modernized_code": deterministic_code,
                "required_imports": [],
                "used_deterministic_rule": True,
            }

    if state["iteration_count"] == 0:
        exemplar_block = _format_exemplar_block(state.get("exemplar_original"), state.get("exemplar_modernized"))
        messages = [
            SystemMessage(content=_with_recipe(handler.refactor_system_prompt, recipe_instruction)),
            HumanMessage(content=f"{exemplar_block}{context_block}{state['original_code']}"),
        ]
    else:
        messages = [
            SystemMessage(content=_with_recipe(handler.fix_system_prompt, recipe_instruction)),
            HumanMessage(
                content=(
                    f"{context_block}"
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
        # Explicitly False here (not left as whatever a PRIOR failed
        # iteration set it to) — this attempt's winning candidate, if
        # any, is LLM-generated, not the deterministic rule's, even if
        # iteration 0 tried the deterministic path first and it failed.
        "used_deterministic_rule": False,
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

    for module in required_imports:
        if check_requires_resolvable(state["language"], module) is False:
            return {
                "status": "failed",
                "compiler_stderr": (
                    f"REQUIRES: {module} names a third-party package, but "
                    f"this sandbox has no network access and never installs "
                    f"dependencies — only modules built into the {state['language']} "
                    f"runtime itself can ever be used here. Rewrite the "
                    f"function without this dependency, using only the "
                    f"standard library/built-in modules."
                ),
            }

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

    # Adversarial counterexample search: ask the model to actively try to
    # find an input where the ORIGINAL and this CANDIDATE diverge, given
    # both versions — a sharper tool than the probes above, which only
    # ever saw the original and picked examples without knowing what the
    # modernization actually changed. Runs once, after every other check
    # already passed (no point spending an extra LLM call + two sandbox
    # round-trips on a candidate about to be rejected anyway), and only
    # for languages where appending a probe is safe (same gate
    # _capture_function_probes uses — see LanguageHandler.
    # supports_function_probe for why cpp/java are excluded).
    if result["status"] == "success" and handler.supports_function_probe:
        adversarial_snippet = generate_adversarial_probe(state["language"], state["original_code"], code)
        if adversarial_snippet is not None:
            original_probe = state["full_source"] + b"\n" + adversarial_snippet.encode("utf-8") + b"\n"
            original_result = verify(
                original_probe.decode("utf-8"), filename=handler.sandbox_filename, run_cmd=handler.run_command(),
            )
            if original_result["status"] == "success":
                # Only meaningful if the ORIGINAL actually ran cleanly
                # with this input — if the model's own counterexample
                # doesn't even run against the original (bad guess, or a
                # local-variable-only reference like the probe-capture
                # path already guards against), there's no baseline to
                # compare the candidate against, so this check is
                # skipped rather than treated as a pass OR a failure.
                candidate_probe = candidate_file + b"\n" + adversarial_snippet.encode("utf-8") + b"\n"
                candidate_result = verify(
                    candidate_probe.decode("utf-8"), filename=handler.sandbox_filename, run_cmd=handler.run_command(),
                )
                if candidate_result["status"] != "success" or candidate_result["stdout"] != original_result["stdout"]:
                    return {
                        "status": "failed",
                        "compiler_stderr": (
                            "An adversarially-chosen input (picked by comparing your "
                            "modernized version against the original) produces a "
                            "DIFFERENT result on the modernized version than on the "
                            "original — this is a real behavioral difference, not a "
                            "false positive.\n\n"
                            f"Input used: {adversarial_snippet}\n\n"
                            f"Original produces: {original_result['stdout']!r}\n\n"
                            f"Modernized produces: "
                            f"{candidate_result.get('stdout', candidate_result.get('stderr'))!r}\n\n"
                            "Fix the function so it produces IDENTICAL results for "
                            "this input too."
                        ),
                    }

    # Property-based equivalence testing (Python only — see
    # property_testing.py's module docstring for exactly when this
    # applies): samples across the ENTIRE input space a fully type-
    # hinted parameter list admits, not just the specific examples the
    # probes/adversarial-search above happened to try. Complementary,
    # not redundant — research on testing LLM-generated code found
    # property-based and example-based approaches each catch a majority
    # of bugs independently, but MORE together than either alone.
    if result["status"] == "success" and state["language"] == "python":
        property_script = property_testing.generate_property_test(state["original_code"], code)
        if property_script is not None:
            property_result = verify(
                property_script, filename="property_test.py", run_cmd="python3 property_test.py",
            )
            if property_result["status"] != "success":
                return {
                    "status": "failed",
                    "compiler_stderr": (
                        "Property-based testing (Hypothesis, sampling across the full "
                        "input space your type hints admit) found an input where the "
                        "modernized function disagrees with the original.\n\n"
                        f"{property_result['stderr']}\n\n"
                        "Fix the function so it produces IDENTICAL results (including "
                        "raising the same kind of exception) for every input its type "
                        "hints allow, not just the specific cases already checked above."
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


def _select_reviewer_llm(used_escalation: bool):
    """Pick a model for assess_risk() that, wherever possible, is NOT the
    same one that wrote the code being reviewed — a model grading its own
    output is a measurably weak checker (it favors its own generative
    patterns), independent of how strong that model is. Preference order:
    1. REVIEWER_MODEL, if explicitly configured — always wins, since it's
       an explicit request for a specific reviewer.
    2. Otherwise, reuse whichever of {base llm, escalation_llm} did NOT
       write this candidate, when escalation_llm is configured at all —
       genuinely different weights, zero new setup for anyone who already
       set ESCALATION_MODEL.
    3. Otherwise (no escalation_llm configured, no REVIEWER_MODEL), fall
       back to the base model — the only one available. Same as this
       project's original behavior; nothing regresses for a zero-config
       user."""
    if reviewer_llm is not None:
        return reviewer_llm
    if escalation_llm is not None:
        return llm if used_escalation else escalation_llm
    return llm


_PUNT_SYSTEM_PROMPT = """You are about to modernize ONE function. Before
attempting it, honestly assess whether you're likely to be able to
rewrite it to modern idioms WITHOUT changing its observable behavior.

Answer with EXACTLY two lines:
PUNT: yes
<one short sentence why — what makes this risky or unclear>

or:

PUNT: no
<one short sentence why — what makes you confident>

Answer PUNT: yes if the function does something you find genuinely
ambiguous or error-prone to reproduce: relies on subtle implicit
behavior of the ORIGINAL language/runtime (implicit type coercion,
undefined-but-relied-upon ordering, a quirky legacy API whose exact
semantics you're unsure of), mixes several unrelated concerns in a way
that's hard to modernize piece-by-piece without risking a behavior
change, or is simply too short/context-free to know what's actually
"legacy" about it versus already fine as-is.

Answer PUNT: no if you're confident you understand exactly what this
function does and how to modernize it while preserving that behavior
exactly — this should be the common case for ordinary, self-contained
legacy code (e.g. missing type hints, old-style string formatting, var
instead of let/const, callback style instead of async/await)."""

_PUNT_RE = re.compile(r"PUNT:\s*(yes|no)", re.IGNORECASE)


def assess_punt(language: str, original_code: str) -> tuple[bool, str]:
    """A lightweight, PRE-attempt confidence check — the mirror image of
    assess_risk (which runs post-success and asks 'was this hard to
    verify'), this runs BEFORE any rewrite attempt and asks 'am I
    confident I can do this correctly at all'. Opt-in (see
    agents/graph.py:modernize's punt_check_enabled) because it costs one
    extra LLM call per chunk to buy the ability to skip chunks the model
    itself doubts, saving the (usually larger) cost of a full best-of-N
    attempt + sandbox verification cycle that was likely to fail anyway.

    Deliberately asks the BASE model (not a reviewer/escalation model,
    unlike assess_risk) — this is the model's own honest self-assessment
    of whether IT can do the job, which is a different question than
    'is a second opinion needed', so self-review concerns don't apply
    the same way here."""
    messages = [
        SystemMessage(content=_PUNT_SYSTEM_PROMPT),
        HumanMessage(content=f"Language: {language}\n\n{original_code}"),
    ]
    response = _invoke_llm_with_retry(llm, messages)
    text = response.content.strip()
    match = _PUNT_RE.search(text)
    punt_flag = bool(match) and match.group(1).lower() == "yes"
    return punt_flag, text


def assess_risk(modernized_code: str, used_escalation: bool = False) -> tuple[bool, str]:
    """One extra LLM call, made only after a chunk has already passed
    structural validation, sandbox compile/run, AND the stdout-equivalence
    check. Those checks prove "this specific run behaved the same" — they
    can't prove "this function can never behave differently," which matters
    for anything with side effects or non-determinism. Reviewed by a
    DIFFERENT model than whichever one wrote modernized_code wherever
    possible (see _select_reviewer_llm) — self-review is a known weak
    checker regardless of model strength, which is a separate concern
    from escalating to a stronger model for difficulty alone."""
    reviewer = _select_reviewer_llm(used_escalation)
    messages = [
        SystemMessage(content=_RISK_SYSTEM_PROMPT),
        HumanMessage(content=modernized_code),
    ]
    response = _invoke_llm_with_retry(reviewer, messages)
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


_MUTATION_SYSTEM_PROMPT = """You will be given ONE function that a human
reviewer trusts is correct. Produce a SUBTLY modified version that
introduces exactly one realistic bug — the kind a careless refactor
might introduce — while staying syntactically valid and superficially
plausible.

Good mutations (pick ONE): flip a comparison operator (== vs !=, < vs
<=, > vs >=); swap + for - or * for /; off-by-one a loop bound, index,
or numeric literal; negate a boolean condition; use the wrong variable
in one place where a similarly-named one exists; drop or duplicate one
statement.

Do NOT change the function's name, parameters, or overall structure —
only introduce the one subtle behavioral bug, somewhere it would
plausibly slip past a quick read.

Respond with ONLY the mutated function. No markdown fences, no
commentary, no explanation, no diff — just the complete rewritten
function body."""


def generate_mutant(language: str, function_code: str) -> tuple[str, list[str]] | None:
    """Ask the model for a deliberately, subtly wrong variant of code
    that already passed every check. Returns (mutant_code,
    required_imports) — required_imports uses the same REQUIRES-marker
    convention as refactorer_node's output, extracted the same way, in
    case the mutation happens to need one (unlikely given the prompt,
    but handled for correctness rather than assumed away). Returns None
    if the model can't/won't produce a plausible, DIFFERENT mutant —
    callers must treat that as "no mutant available to test with," not
    an error, same contract as generate_probes returning []."""
    messages = [
        SystemMessage(content=_MUTATION_SYSTEM_PROMPT),
        HumanMessage(content=function_code),
    ]
    response = _invoke_llm_with_retry(llm, messages)
    fenced = _strip_markdown_fence(response.content)
    mutant_code, required_imports = _extract_required_imports(fenced)
    mutant_code = mutant_code.strip()
    if not mutant_code or mutant_code == function_code.strip():
        return None
    return mutant_code, required_imports


def check_mutation_confidence(
    handler, state: AgentState, modernized_code: str, required_imports: list[str]
) -> tuple[bool, str]:
    """Deliberately mutate the ALREADY-VERIFIED modernized_code and run
    the mutant through the EXACT same verification (_verify_candidate —
    structural validation, sandbox compile/run, baseline comparison,
    every probe, determinism recheck) that modernized_code itself just
    passed. If the mutant ALSO passes, that's a signal about the
    STRENGTH of THIS chunk's specific checks, not about modernized_code
    — a version we KNOW is behaviorally broken slipped past baseline/
    probe comparison, meaning "success" here deserves less confidence
    than usual (too few probes, or probes that don't happen to exercise
    the mutated code path). This is the "verify your own verification"
    counterpart to check_requires_resolvable and the probe/determinism
    checks already in _verify_candidate — same spirit as classic
    mutation testing (deliberately break the code, confirm your checks
    catch it), adapted to a run that generates its own checks on the
    fly instead of reusing a pre-existing test suite.

    Post-success only: no point spending an extra LLM call + sandbox
    round-trip mutating a chunk that was already rejected. Flags,
    doesn't block — mirrors assess_risk/scan_security's design: a
    failed mutation check says "trust this result a bit less," never
    "discard a result that already passed rigorous verification.\""""
    mutant = generate_mutant(state["language"], modernized_code)
    if mutant is None:
        return False, ""
    mutant_code, mutant_imports = mutant
    merged_imports = list(required_imports)
    for m in mutant_imports:
        if m not in merged_imports:
            merged_imports.append(m)

    result = _verify_candidate(handler, state, mutant_code, merged_imports)
    if result["status"] == "success":
        return True, (
            "A deliberately broken variant of this function passed the "
            "same verification (baseline/probe checks) that the real "
            "modernization did — this chunk's specific checks may not "
            "have enough coverage to catch a subtle regression here."
        )
    return False, ""


def fallback_node(state: AgentState) -> AgentState:
    return {**state, "status": "gave_up"}


_PROBE_LANGUAGE_LABELS = {
    "python": ("Python", "print(...)"),
    "javascript": ("JavaScript", "console.log(...)"),
    "typescript": ("TypeScript", "console.log(...)"),
    "php": ("PHP", "echo ...;"),
}

_PROBE_SYSTEM_PROMPT_TEMPLATE = """You will be given ONE {label} function
and a target number of examples. Write that many short snippets, each on
its own line, that call this function with DIFFERENT example arguments
and print/output the result — so its behavior can be compared across a
RANGE of inputs, not just one. A single example proves almost nothing
about edge cases; several varied ones is a real check.

Rules:
- Each line is a complete, independent call+print statement — nothing
  else on that line.
- Call the function using its exact name as shown.
- Make the lines meaningfully different from each other: cover a typical
  case, an edge case (empty string / zero / negative / boundary value),
  and another distinct case if more are requested — don't just change
  one digit each time.
- Print/output the result using this language's normal mechanism: {print_call}
- Do NOT redefine the function. Do NOT add imports or requires. Do NOT add
  explanatory text, numbering, or markdown fences.
- If the function needs objects/types too complex to construct from
  scratch (e.g. it takes a database connection), respond with EXACTLY:
  PROBE: SKIP

Respond with ONLY that many lines (or PROBE: SKIP), nothing else."""


def generate_probes(language: str, function_code: str, count: int = 3) -> list[str]:
    """Ask the model for MULTIPLE diverse 'call this function, print the
    result' snippets instead of just one. A single hand-picked example
    proves almost nothing about edge cases (empty/zero/negative/boundary
    values) it never exercises — this directly targets that gap, in the
    same spirit as differential-fuzzing approaches to verifying LLM code
    refactorings (generate many inputs, compare outputs across all of
    them, not just one). Returns [] if the model can't/won't produce
    any; callers must treat that as "no synthesized probes available,"
    not an error.

    `count` is deliberately passed via the HUMAN message, not templated
    into the system prompt: the system prompt is otherwise 100% static
    per language, which is exactly the shape that lets a caching-aware
    LLM provider (Claude, GPT — not Ollama, which doesn't bill by token)
    reuse the cached prefix across every probe-generation call for that
    language. Templating a per-call value (count varies 1-3 depending on
    how many real call sites were already found) into the system prompt
    would silently fragment that cache into one entry per count value."""
    label, print_call = _PROBE_LANGUAGE_LABELS[language]
    prompt = _PROBE_SYSTEM_PROMPT_TEMPLATE.format(label=label, print_call=print_call)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Generate exactly {count} example(s) for this function:\n\n{function_code}"),
    ]
    response = _invoke_llm_with_retry(llm, messages)
    text = _strip_markdown_fence(response.content).strip()
    if not text or "SKIP" in text.upper():
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:count]


_ADVERSARIAL_SYSTEM_PROMPT_TEMPLATE = """You will be given TWO versions of
the same {label} function: the ORIGINAL (legacy) version and a MODERNIZED
version that is supposed to behave IDENTICALLY for every input. Your job
is to actively try to DISPROVE that — find ONE input that would make them
produce different results, if such an input exists.

Think adversarially about what the modernization might have subtly
changed: type coercion differences, boundary values (empty, zero,
negative, very large, None/null), order-of-operations changes, off-by-one
errors, truncation/rounding differences, or a semantic gap between the
old idiom and the new one used to replace it (e.g. does the new string
method handle an empty string the same way the old one did?).

Respond with EXACTLY one line: a single call+print statement, using the
function's exact name, for the input you believe is MOST likely to
reveal a difference: {print_call}

If you genuinely cannot think of any input that would reveal a
difference, respond with EXACTLY: PROBE: SKIP"""


def generate_adversarial_probe(language: str, original_code: str, modernized_code: str) -> str | None:
    """Ask the model to actively search for an input that would make the
    ORIGINAL and MODERNIZED versions diverge — unlike generate_probes
    (which only sees the original and picks 'diverse' examples) or
    check_mutation_confidence (which mutates the code, not the input),
    this sees BOTH versions and is explicitly prompted to find a
    counterexample, the same framing used in recent program-equivalence
    research (disprove equivalence by construction, not by sampling).
    Returns a probe snippet, or None if unsupported for this language or
    the model couldn't/wouldn't produce one — callers treat None as 'no
    adversarial check available,' not an error, same contract as
    generate_probes."""
    if language not in _PROBE_LANGUAGE_LABELS:
        return None
    label, print_call = _PROBE_LANGUAGE_LABELS[language]
    prompt = _ADVERSARIAL_SYSTEM_PROMPT_TEMPLATE.format(label=label, print_call=print_call)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"ORIGINAL:\n{original_code}\n\nMODERNIZED:\n{modernized_code}"),
    ]
    response = _invoke_llm_with_retry(llm, messages)
    text = _strip_markdown_fence(response.content).strip()
    if not text or "SKIP" in text.upper():
        return None
    # Defensive: take only the first non-empty line even if the model
    # wrote more than the requested single line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None


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
