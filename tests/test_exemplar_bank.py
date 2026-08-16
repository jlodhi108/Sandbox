import os
import tempfile
from unittest.mock import patch

import exemplar_bank
import embeddings


def _reset_cache():
    exemplar_bank._cache = None


def test_record_is_noop_when_embeddings_disabled():
    _reset_cache()
    with patch.object(embeddings, "is_enabled", return_value=False):
        exemplar_bank.record("python", "def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")
    assert exemplar_bank._get_cache() == {}


def test_find_best_exemplar_returns_none_when_embeddings_disabled():
    _reset_cache()
    with patch.object(embeddings, "is_enabled", return_value=False):
        assert exemplar_bank.find_best_exemplar("python", "def g(x, y): return x+y") is None


def test_find_best_exemplar_returns_none_with_no_stored_exemplars():
    _reset_cache()
    with patch.object(embeddings, "is_enabled", return_value=True):
        assert exemplar_bank.find_best_exemplar("python", "def g(x, y): return x+y") is None


def test_record_and_find_round_trip():
    _reset_cache()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")
        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path), \
             patch.object(embeddings, "is_enabled", return_value=True), \
             patch.object(embeddings, "rank_by_relevance", side_effect=lambda q, c: c):
            exemplar_bank.record("python", "def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")
            result = exemplar_bank.find_best_exemplar("python", "def g(x, y): return x+y")

    assert result == {"original": "def f(a, b): return a+b", "modernized": "def f(a: int, b: int) -> int: return a+b"}


def test_record_deduplicates_identical_pairs():
    _reset_cache()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")
        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path), \
             patch.object(embeddings, "is_enabled", return_value=True):
            exemplar_bank.record("python", "def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")
            exemplar_bank.record("python", "def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")

    assert len(exemplar_bank._get_cache()["python"]) == 1


def test_record_caps_at_max_exemplars_per_language_dropping_oldest():
    _reset_cache()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")
        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path), \
             patch.object(embeddings, "is_enabled", return_value=True), \
             patch.object(exemplar_bank, "MAX_EXEMPLARS_PER_LANGUAGE", 3):
            for i in range(5):
                exemplar_bank.record("python", f"def f{i}(): pass", f"def f{i}() -> None: pass")

    entries = exemplar_bank._get_cache()["python"]
    assert len(entries) == 3
    # oldest (f0, f1) dropped, newest (f2, f3, f4) kept
    originals = [e["original"] for e in entries]
    assert originals == ["def f2(): pass", "def f3(): pass", "def f4(): pass"]


def test_record_persists_to_disk():
    _reset_cache()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")
        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path), \
             patch.object(embeddings, "is_enabled", return_value=True):
            exemplar_bank.record("python", "def f(a, b): return a+b", "def f(a: int, b: int) -> int: return a+b")

        assert os.path.isfile(path)
        _reset_cache()
        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path):
            loaded = exemplar_bank._get_cache()
    assert loaded["python"][0]["original"] == "def f(a, b): return a+b"


def test_load_recovers_from_corrupt_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert exemplar_bank._load(path) == {}


def test_find_best_exemplar_picks_most_similar_via_ranking():
    _reset_cache()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "exemplars.json")

        def fake_rank(query, candidates):
            # pretend the SECOND candidate is always most relevant
            return [candidates[1], candidates[0]] if len(candidates) > 1 else candidates

        with patch.object(exemplar_bank, "DEFAULT_EXEMPLAR_PATH", path), \
             patch.object(embeddings, "is_enabled", return_value=True), \
             patch.object(embeddings, "rank_by_relevance", side_effect=fake_rank):
            exemplar_bank.record("python", "def a(): pass", "def a() -> None: pass")
            exemplar_bank.record("python", "def b(): pass", "def b() -> None: pass")
            result = exemplar_bank.find_best_exemplar("python", "def query(): pass")

    assert result["original"] == "def b(): pass"
