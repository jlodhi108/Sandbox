import pytest

from llm_budget import LLMBudget, BudgetExceededError


def test_check_never_raises_when_unlimited():
    budget = LLMBudget()  # max_calls=None
    for _ in range(1000):
        budget.check()
        budget.record("qwen2.5-coder:7b", None)
    assert budget.total_calls == 1000


def test_check_raises_once_max_calls_reached():
    budget = LLMBudget(max_calls=3)
    for _ in range(3):
        budget.check()
        budget.record("qwen2.5-coder:7b", None)
    with pytest.raises(BudgetExceededError):
        budget.check()


def test_is_exceeded_matches_check_but_does_not_raise():
    budget = LLMBudget(max_calls=2)
    assert budget.is_exceeded() is False
    budget.record("qwen2.5-coder:7b", None)
    budget.record("qwen2.5-coder:7b", None)
    assert budget.is_exceeded() is True
    with pytest.raises(BudgetExceededError):
        budget.check()


def test_record_accumulates_tokens_and_per_model_counts():
    budget = LLMBudget()
    budget.record("qwen2.5-coder:7b", {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
    budget.record("qwen2.5-coder:7b", {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60})
    budget.record("qwen2.5-coder:32b", {"input_tokens": 200, "output_tokens": 40, "total_tokens": 240})
    summary = budget.summary()
    assert summary["total_calls"] == 3
    assert summary["total_input_tokens"] == 350
    assert summary["total_output_tokens"] == 70
    assert summary["calls_by_model"] == {"qwen2.5-coder:7b": 2, "qwen2.5-coder:32b": 1}


def test_record_tolerates_missing_usage_metadata():
    # Not every backend/response necessarily includes usage_metadata —
    # call counting must not depend on token data being present.
    budget = LLMBudget()
    budget.record("qwen2.5-coder:7b", None)
    summary = budget.summary()
    assert summary["total_calls"] == 1
    assert summary["total_input_tokens"] == 0


def test_record_ignores_malformed_usage_metadata_without_crashing():
    # A test mocking an LLM response with a bare MagicMock() means
    # getattr(response, "usage_metadata", None) returns ANOTHER
    # MagicMock (truthy, auto-generates a .get() that also returns a
    # MagicMock) rather than a real dict or None — this must be quietly
    # ignored, not accumulated, or total_input_tokens ends up holding a
    # non-int value that breaks JSON report serialization later.
    from unittest.mock import MagicMock
    import json

    budget = LLMBudget()
    budget.record("fake-model", MagicMock())  # not a dict at all
    budget.record("fake-model", {"input_tokens": "not-a-number", "output_tokens": None})
    summary = budget.summary()
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 0
    assert summary["total_output_tokens"] == 0
    json.dumps(summary)  # must not raise


def test_reset_clears_counts_and_installs_new_ceiling():
    budget = LLMBudget(max_calls=5)
    budget.record("qwen2.5-coder:7b", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    budget.reset(max_calls=100)
    summary = budget.summary()
    assert summary["total_calls"] == 0
    assert summary["total_input_tokens"] == 0
    assert summary["calls_by_model"] == {}
    assert summary["max_calls"] == 100


def test_record_coerces_non_string_model_name_to_a_serializable_key():
    # A test mocking the LLM instance itself (not just its response)
    # means llm_instance.model is a MagicMock too, not a real string —
    # calls_by_model's keys must stay JSON-serializable regardless.
    from unittest.mock import MagicMock
    import json

    budget = LLMBudget()
    budget.record(MagicMock(), None)
    summary = budget.summary()
    assert summary["total_calls"] == 1
    (key,) = summary["calls_by_model"].keys()
    assert isinstance(key, str)
    json.dumps(summary)  # must not raise


def test_reset_with_no_argument_means_unlimited():
    budget = LLMBudget(max_calls=5)
    budget.reset()
    budget.check()  # must not raise — no cap after a bare reset()
    assert budget.summary()["max_calls"] is None
