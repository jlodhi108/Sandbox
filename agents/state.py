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
    baseline_stdout: str | None  # stdout of the file BEFORE this chunk's
                                  # modernization, captured once up front.
                                  # None if the original itself didn't run
                                  # cleanly (nothing to compare against).
    probe_snippet: str | None       # "call this function, print result"
                                     # snippet, generated once up front.
                                     # None if unsupported for this
                                     # language or generation failed.
    probe_baseline_stdout: str | None  # probe's output against the
                                        # ORIGINAL function, to compare
                                        # the modernized version against.
    used_escalation: bool   # True once any attempt on this chunk used the
                             # stronger escalation model, not just the
                             # default — surfaced in the run report
    risk_flag: bool         # set post-graph, only on success: does this
                             # change touch something a stdout-diff test
                             # can't prove is equivalent (I/O, globals,
                             # randomness, timing)?
    risk_reason: str
    compiler_stderr: str
    iteration_count: int
    status: str  # "pending" | "success" | "failed"
    max_iterations: int
