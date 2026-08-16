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
    modernized_code: str    # the WINNING candidate's code — either the
                             # only one tried, or whichever of
                             # candidate_codes actually passed verification
    required_imports: list[str]  # the winning candidate's REQUIRES markers
    candidate_codes: list[dict]  # [{"code": str, "required_imports":
                                  # list[str]}, ...] — set by
                                  # refactorer_node each call. More than
                                  # one entry only on iteration 0 (see
                                  # BEST_OF_N_ON_FIRST_ATTEMPT);
                                  # verifier_node tries each in order and
                                  # keeps the first that passes.
    baseline_stdout: str | None  # stdout of the file BEFORE this chunk's
                                  # modernization, captured once up front.
                                  # None if the original itself didn't run
                                  # cleanly (nothing to compare against).
    probes: list[dict]      # [{"snippet": str, "baseline_stdout": str}, ...]
                             # captured once up front, against the ORIGINAL
                             # function — real call sites found in the
                             # codebase plus LLM-synthesized diverse
                             # examples. Empty list if unsupported for
                             # this language or none could be verified.
    used_escalation: bool   # True once any attempt on this chunk used the
                             # stronger escalation model, not just the
                             # default — surfaced in the run report
    risk_flag: bool         # set post-graph, only on success: does this
                             # change touch something a stdout-diff test
                             # can't prove is equivalent (I/O, globals,
                             # randomness, timing)?
    risk_reason: str
    security_flag: bool     # set post-graph, only on success: did a
                             # static-analysis security scan (semgrep,
                             # local rules) find anything in the
                             # modernized code — a DIFFERENT gap than
                             # risk_flag, since behavioral equivalence
                             # says nothing about newly introduced
                             # vulnerabilities.
    security_findings: list[dict]  # [{"rule_id", "line", "message"}, ...]
    mutation_confidence_flag: bool  # set post-graph, only on success: a
                                      # deliberately-broken mutant of this
                                      # chunk was generated and run through
                                      # the SAME baseline+probe checks that
                                      # just passed — True means the checks
                                      # did NOT catch the known-broken
                                      # mutant, i.e. this chunk's specific
                                      # verification had no real "bite" and
                                      # its "success" deserves less trust
                                      # than usual. A DIFFERENT gap than
                                      # risk_flag/security_flag: this is
                                      # about the STRENGTH of the check
                                      # itself, not a property of the code.
    mutation_confidence_reason: str
    review_thread_id: str | None  # set post-graph, only when interactive=True
                                    # AND at least one flag above is set: the
                                    # id a caller passes to
                                    # agents.review_graph.resume_review() to
                                    # approve/reject this chunk. status is
                                    # "awaiting_review" (not "success") while
                                    # this is set and unresolved — see
                                    # modernize()'s interactive handling.
    compiler_stderr: str
    iteration_count: int
    status: str  # "pending" | "success" | "failed" | "awaiting_review"
    max_iterations: int
    recipe_instruction: str | None  # extra guidance appended to the refactor/
                                      # fix system prompts, e.g. "convert
                                      # callback-style functions to async/await
                                      # only — leave everything else as-is."
                                      # None (the default) changes nothing
                                      # about the prompt. Set once per run
                                      # from .modernizer.toml's [recipes.<name>]
                                      # via --recipe, not per-chunk — passed
                                      # through state (not a module-level
                                      # global) so concurrent --workers runs
                                      # of DIFFERENT files can't race on it.
