import json
import os

DEFAULT_HISTORY_PATH = ".modernizer_history.json"


def load_history(path: str = DEFAULT_HISTORY_PATH) -> dict:
    """Per-language cumulative {chunks_succeeded, chunks_attempted} across
    every past run. Returns {} if no history file exists yet — a language
    with no history is, correctly, not yet eligible for auto-PR (see
    is_eligible): "no track record" and "bad track record" must both
    fail closed, not open."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_history(history: dict, path: str = DEFAULT_HISTORY_PATH) -> None:
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def record_run(history: dict, language: str, succeeded: int, attempted: int) -> dict:
    """Fold one run's per-language counts into history. Pure — returns a
    NEW dict rather than mutating in place, and doesn't write to disk
    (caller decides when to persist). `attempted` should exclude
    already-modern chunks (skipped before ever reaching the LLM/sandbox
    pipeline) — only chunks that actually went through verification
    count toward track record either way."""
    updated = {k: dict(v) for k, v in history.items()}
    entry = updated.setdefault(language, {"chunks_succeeded": 0, "chunks_attempted": 0})
    entry["chunks_succeeded"] += succeeded
    entry["chunks_attempted"] += attempted
    return updated


def is_eligible(
    language: str, history: dict, min_chunks: int = 5, min_success_rate: float = 0.8,
) -> tuple[bool, str]:
    """Whether `language` has a proven-enough track record (from PRIOR
    runs — the CURRENT run's own results are never included, which would
    let a run bootstrap its own eligibility) to trust for automatic PR
    creation. Matches current industry guidance for autonomous coding
    agents: start with narrow permissions and widen only as track record
    proves out per-codebase/language, not a fixed global trust level.
    Fails closed — a language with zero history is treated the same as
    one with a bad history, not silently allowed through."""
    entry = history.get(language, {"chunks_succeeded": 0, "chunks_attempted": 0})
    attempted = entry["chunks_attempted"]
    if attempted < min_chunks:
        return False, (
            f"only {attempted}/{min_chunks} historical chunk(s) attempted for "
            f"'{language}' — not enough track record yet for auto-PR"
        )
    rate = entry["chunks_succeeded"] / attempted
    if rate < min_success_rate:
        return False, (
            f"'{language}' historical success rate {rate:.0%} "
            f"({entry['chunks_succeeded']}/{attempted}) is below the "
            f"{min_success_rate:.0%} threshold required for auto-PR"
        )
    return True, (
        f"'{language}' track record: {entry['chunks_succeeded']}/{attempted} "
        f"({rate:.0%}) — eligible for auto-PR"
    )
