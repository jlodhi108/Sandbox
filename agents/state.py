from typing import TypedDict


class AgentState(TypedDict):
    language: str           # registry key, e.g. "cpp", "python" — looked
                             # up via languages.get_handler() by nodes/graph
    full_source: bytes      # the ENTIRE original file — needed so the
                             # verifier compiles/runs the whole translation
                             # unit, not the chunk in isolation. Kept as
                             # bytes because chunk_start/chunk_end are BYTE
                             # offsets from Tree-sitter (a `str` would
                             # misalign on any non-ASCII content).
    chunk_start: int
    chunk_end: int
    original_code: str      # the chunk's original text, for LLM reference
    modernized_code: str    # the chunk's current candidate replacement
    required_imports: list[str]  # modules/headers the model requested via
                                  # `REQUIRES: <module>` markers
    compiler_stderr: str
    iteration_count: int
    status: str  # "pending" | "success" | "failed"
    max_iterations: int
