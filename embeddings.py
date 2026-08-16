"""Optional local embedding-based ranking for cross-file context
selection (see agents/graph.py's _extract_context_signatures and
_extract_referenced_type_definitions) — when EMBEDDING_MODEL is set
(e.g. "nomic-embed-text", pulled via `ollama pull nomic-embed-text`
exactly like any other model this project uses), sibling files get
ranked by actual semantic relevance to the chunk being modernized
instead of positional order ("the first N files discover_files()
happened to list"). In a large repo, the first N files alphabetically/
by-walk-order are rarely the most relevant N — this makes the existing
capped scan (MAX_CONTEXT_SIBLING_FILES) actually prioritize what
matters instead of whatever happened to be discovered first.

Off by default — same "unless you explicitly configure a model name"
pattern as ESCALATION_MODEL/REVIEWER_MODEL (agents/nodes.py) — because
it's a genuinely optional enhancement (the existing capped scan already
works, just less precisely at repo scale) that requires an extra model
pull most users haven't done, and an extra Ollama round-trip per
sibling file per run (mitigated by the cache below, but still not
free). Uses langchain_ollama.OllamaEmbeddings — already a dependency
via langchain-ollama, no new package needed."""
import math
import os

from langchain_ollama import OllamaEmbeddings

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL) if EMBEDDING_MODEL else None

# Keyed by raw content bytes, not a hash — content is already in memory
# (these are file sources this run already read), and this avoids ever
# needing to reason about hash collisions for something whose only job
# is a best-effort ranking, not correctness-critical identity.
_embedding_cache: dict[bytes, list[float] | None] = {}


def is_enabled() -> bool:
    return _embedder is not None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed(content: bytes) -> list[float] | None:
    """Cached, best-effort embedding. Returns None (never raises) on any
    failure — Ollama unreachable, embedding model not pulled, empty
    content — so callers degrade to their non-semantic fallback instead
    of a ranking enhancement crashing an otherwise-working
    modernization run."""
    if content in _embedding_cache:
        return _embedding_cache[content]
    result = None
    try:
        decoded = content.decode("utf-8", errors="replace").strip()
        if decoded:
            result = _embedder.embed_query(decoded)
    except Exception:
        result = None
    _embedding_cache[content] = result
    return result


def rank_by_relevance(query_code: str, candidates: list[bytes]) -> list[bytes]:
    """Sort `candidates` (sibling file contents) by semantic similarity
    to query_code (the chunk being modernized), most relevant first.
    Returns candidates in their ORIGINAL order, unchanged, if embeddings
    aren't enabled (EMBEDDING_MODEL unset) or the query itself can't be
    embedded — degrading to positional order (the pre-existing behavior)
    rather than a partially-ranked or crashing result. An individual
    candidate that fails to embed is ranked last (score -1.0, always
    below any real cosine similarity in [-1, 1]) rather than dropped —
    still available, just deprioritized, consistent with this being a
    ranking aid, not a filter."""
    if not is_enabled():
        return candidates
    query_embedding = _embed(query_code.encode("utf-8"))
    if query_embedding is None:
        return candidates

    scored = []
    for candidate in candidates:
        embedding = _embed(candidate)
        score = _cosine_similarity(query_embedding, embedding) if embedding is not None else -1.0
        scored.append((score, candidate))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [candidate for _, candidate in scored]
