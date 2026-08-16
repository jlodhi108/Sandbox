import os
import tempfile

from track_record import load_history, save_history, record_run, is_eligible


def test_load_history_returns_empty_dict_when_missing():
    assert load_history("/nonexistent/.modernizer_history.json") == {}


def test_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "history.json")
        history = {"python": {"chunks_succeeded": 8, "chunks_attempted": 10}}
        save_history(history, path)
        assert load_history(path) == history


def test_save_history_leaves_no_tmp_file_behind():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "history.json")
        save_history({"python": {"chunks_succeeded": 1, "chunks_attempted": 1}}, path)
        assert os.listdir(root) == ["history.json"]


def test_load_history_recovers_from_corrupt_file():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "history.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        assert load_history(path) == {}


def test_record_run_creates_new_language_entry():
    history = {}
    updated = record_run(history, "python", succeeded=3, attempted=4)
    assert updated["python"] == {"chunks_succeeded": 3, "chunks_attempted": 4}
    assert history == {}  # pure — original untouched


def test_record_run_accumulates_into_existing_entry():
    history = {"python": {"chunks_succeeded": 5, "chunks_attempted": 6}}
    updated = record_run(history, "python", succeeded=2, attempted=2)
    assert updated["python"] == {"chunks_succeeded": 7, "chunks_attempted": 8}


def test_is_eligible_fails_closed_with_no_history():
    eligible, reason = is_eligible("python", {})
    assert eligible is False
    assert "not enough track record" in reason


def test_is_eligible_fails_below_min_chunks_even_with_perfect_rate():
    # 3/3 = 100% success, but only 3 attempts — must still fail if the
    # threshold is 5. A tiny sample shouldn't buy trust.
    history = {"python": {"chunks_succeeded": 3, "chunks_attempted": 3}}
    eligible, reason = is_eligible("python", history, min_chunks=5)
    assert eligible is False
    assert "3/5" in reason


def test_is_eligible_fails_below_success_rate_threshold():
    history = {"python": {"chunks_succeeded": 5, "chunks_attempted": 10}}  # 50%
    eligible, reason = is_eligible("python", history, min_chunks=5, min_success_rate=0.8)
    assert eligible is False
    assert "50%" in reason


def test_is_eligible_passes_with_sufficient_proven_track_record():
    history = {"python": {"chunks_succeeded": 9, "chunks_attempted": 10}}  # 90%
    eligible, reason = is_eligible("python", history, min_chunks=5, min_success_rate=0.8)
    assert eligible is True
    assert "90%" in reason


def test_is_eligible_is_per_language_independent():
    history = {
        "python": {"chunks_succeeded": 9, "chunks_attempted": 10},
        "java": {"chunks_succeeded": 1, "chunks_attempted": 10},
    }
    py_eligible, _ = is_eligible("python", history)
    java_eligible, _ = is_eligible("java", history)
    assert py_eligible is True
    assert java_eligible is False
