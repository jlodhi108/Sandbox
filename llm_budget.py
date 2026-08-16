import threading


class BudgetExceededError(RuntimeError):
    """Raised by LLMBudget.check() when the NEXT call would exceed the
    configured ceiling — checked before a call is made, not after, so
    the ceiling bounds calls actually made, not calls attempted."""


class LLMBudget:
    """Tracks LLM calls/tokens for ONE run and optionally enforces a
    hard call ceiling — a circuit breaker for repo-mode runs, where
    best-of-N x retry-iterations x probes x chunks x files can multiply
    fast with nothing capping it otherwise. Thread-safe (a lock guards
    every mutation) since --workers > 1 shares one instance across
    threads; under concurrency the cap is a SOFT limit — several
    in-flight calls can pass check() near-simultaneously before any of
    them increments, so a run can overshoot max_calls by up to
    (worker count) calls. That's an accepted trade-off for a runaway-
    cost circuit breaker, not a hard security boundary."""

    def __init__(self, max_calls: int | None = None):
        self.max_calls = max_calls
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_model: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self, max_calls: int | None = None) -> None:
        """Start tracking a fresh run. Call once at the TRUE top-level
        entry point (the CLI's __main__ block, or an MCP tool function)
        — never inside run_file when it's being called per-file FROM
        run_repo, or a repo-wide budget would silently reset on every
        file instead of accumulating across the whole run."""
        with self._lock:
            self.max_calls = max_calls
            self.total_calls = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.calls_by_model = {}

    def check(self) -> None:
        with self._lock:
            if self._is_exceeded_locked():
                raise BudgetExceededError(
                    f"LLM call budget exceeded: {self.total_calls}/{self.max_calls} "
                    f"calls already made this run. Raise max_llm_calls_per_run in "
                    f".modernizer.toml's [settings] table (or the "
                    f"MAX_LLM_CALLS_PER_RUN env var) to allow more, or investigate "
                    f"why this run needed so many."
                )

    def is_exceeded(self) -> bool:
        """Non-raising version of check(), for callers that want to
        decide NOT to start a new unit of work (e.g. run_repo skipping
        the next file entirely) rather than start it and hit the
        exception partway through."""
        with self._lock:
            return self._is_exceeded_locked()

    def _is_exceeded_locked(self) -> bool:
        return self.max_calls is not None and self.total_calls >= self.max_calls

    def record(self, model_name: str, usage_metadata: dict | None) -> None:
        with self._lock:
            self.total_calls += 1
            # str() defensively: model_name should always already be a
            # plain string (ChatOllama.model is), but a caller passing
            # something else (e.g. a test mock with no real .model
            # attribute) must still produce a JSON-serializable dict key
            # rather than silently corrupting this tracker's state.
            model_name = str(model_name)
            self.calls_by_model[model_name] = self.calls_by_model.get(model_name, 0) + 1
            # Token accounting is best-effort and must never corrupt this
            # tracker's own state (e.g. later breaking JSON report
            # serialization) — only accept a real, plain dict with actual
            # int values; anything else (missing, malformed, or a mocked
            # response in a test whose auto-generated attributes are NOT
            # plain ints) is silently ignored rather than trusted as-is.
            if isinstance(usage_metadata, dict):
                input_tokens = usage_metadata.get("input_tokens")
                output_tokens = usage_metadata.get("output_tokens")
                if isinstance(input_tokens, int):
                    self.total_input_tokens += input_tokens
                if isinstance(output_tokens, int):
                    self.total_output_tokens += output_tokens

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "calls_by_model": dict(self.calls_by_model),
                "max_calls": self.max_calls,
            }
