import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import AgentState
from sandbox.verifier import verify
from languages import get_handler_by_name

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0)

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

    response = llm.invoke(messages)
    fenced = _strip_markdown_fence(response.content)
    clean_code, required_imports = _extract_required_imports(fenced)
    return {
        **state,
        "modernized_code": clean_code,
        "required_imports": required_imports,
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
    return {
        **state,
        "compiler_stderr": result["stderr"],
        "status": result["status"],
        "iteration_count": state["iteration_count"] + 1,
    }


def fallback_node(state: AgentState) -> AgentState:
    return {**state, "status": "gave_up"}
