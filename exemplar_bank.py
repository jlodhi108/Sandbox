"""A small, per-language library of ACTUAL successful (original ->
modernized) diff pairs from past runs — used to give the refactor
prompt a real, in-context example of "here's what a GOOD modernization
of similar legacy code from THIS project's own proven track record
looks like," rather than relying purely on the base system prompt's
generic instructions. In-context learning from this project's own
results, not a hand-picked hardcoded sample.

Hard-gated on embeddings.is_enabled() for BOTH recording and retrieval,
not just retrieval: without a real relevance ranking (see embeddings.py),
showing an ARBITRARY past exemplar risks steering the model toward an
unrelated pattern, which is worse than showing none at all — and there's
no point accumulating exemplars on disk that retrieval can never
usefully select between either. Both halves of this feature only ever
activate together, when EMBEDDING_MODEL is configured.

Storage: a local JSON file, same footing as track_record.py's
.modernizer_history.json — per-machine, accumulated run state, never
meant to be committed (see .gitignore). Self-contained module-level
state (lazy-loaded, written through on every recording) rather than
main.py orchestrating load/save the way it does for track_record.py's
_history — agents/graph.py (which records/retrieves exemplars) has no
business reaching into main.py's module state, and recording happens
rarely enough (once per successfully modernized chunk, not once per
LLM call) that a write-through is a small, simple cost rather than
something worth batching."""
import json
import os

import embeddings

DEFAULT_EXEMPLAR_PATH = ".modernizer_exemplars.json"
MAX_EXEMPLARS_PER_LANGUAGE = 50

_cache: dict[str, list[dict]] | None = None


def _load(path: str) -> dict[str, list[dict]]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"    (warning: {path} is corrupt — starting with an empty exemplar bank)")
        return {}


def _save(exemplars: dict[str, list[dict]], path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(exemplars, f, indent=2)
    os.replace(tmp_path, path)


def _get_cache() -> dict[str, list[dict]]:
    # Deliberately reads the module-level DEFAULT_EXEMPLAR_PATH by NAME
    # here rather than via a function default parameter (`path:
    # str = DEFAULT_EXEMPLAR_PATH`) — a default parameter's value is
    # bound ONCE at function-definition time, so tests patching this
    # module's DEFAULT_EXEMPLAR_PATH attribute (to redirect I/O to a
    # temp file) would silently have no effect on it. A plain global
    # lookup inside the function body is resolved fresh on every call.
    global _cache
    if _cache is None:
        _cache = _load(DEFAULT_EXEMPLAR_PATH)
    return _cache


def record(language: str, original_code: str, modernized_code: str) -> None:
    """Persist one successful (original, modernized) pair for `language`
    — a no-op if embeddings aren't enabled (see this module's docstring
    for why recording and retrieval share one gate). Deduplicates
    identical (original, modernized) pairs (a chunk modernized the same
    way across multiple runs shouldn't multiply entries) and caps at
    MAX_EXEMPLARS_PER_LANGUAGE, dropping the OLDEST when exceeded — a
    rolling window of recent proven successes, not an ever-growing
    archive."""
    if not embeddings.is_enabled():
        return
    cache = _get_cache()
    entries = cache.setdefault(language, [])
    pair = {"original": original_code, "modernized": modernized_code}
    if pair in entries:
        return
    entries.append(pair)
    if len(entries) > MAX_EXEMPLARS_PER_LANGUAGE:
        del entries[: len(entries) - MAX_EXEMPLARS_PER_LANGUAGE]
    _save(cache, DEFAULT_EXEMPLAR_PATH)


def find_best_exemplar(language: str, query_code: str) -> dict | None:
    """The single most semantically-similar past success for `language`
    to query_code (the chunk about to be modernized), or None if
    embeddings aren't enabled or there are no stored exemplars for this
    language yet. Never returns the exemplar for query_code AGAINST
    ITSELF — irrelevant in practice (a chunk about to be modernized was,
    by definition, never already recorded as a past success under the
    exact same original text), not specially guarded against."""
    if not embeddings.is_enabled():
        return None
    entries = _get_cache().get(language, [])
    if not entries:
        return None
    candidates = [entry["original"].encode("utf-8") for entry in entries]
    ranked = embeddings.rank_by_relevance(query_code, candidates)
    if not ranked:
        return None
    best_original = ranked[0]
    for entry in entries:
        if entry["original"].encode("utf-8") == best_original:
            return entry
    return None
