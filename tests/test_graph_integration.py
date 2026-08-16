"""Integration tests for the LangGraph wiring itself: retry loop,
escalation switching, risk-assessment hookup, and probe-check hookup.

Unlike tests/test_nodes.py (which tests each node function in isolation
with mocks), these run the REAL compiled graph via agents.graph.modernize()
end to end — real state transitions, real router decisions, real node
composition — with only the LLM and the sandbox verify() faked out. This
is what actually proves the pieces wire together correctly; the unit
tests only prove each piece works alone. No Docker or Ollama needed, so
this runs in CI.

Patch targets, and why: a function's global lookups always resolve
against the module it was DEFINED in, not the module that imported it.
refactorer_node/verifier_node/assess_risk/generate_probe are all defined
in agents.nodes, so patching agents.nodes.llm / agents.nodes.verify
covers all of them regardless of how agents.graph imported them. But
_capture_baseline_stdout, _capture_function_probe, and the assess_risk/
generate_probe CALL SITES inside modernize() are defined in agents.graph,
which did `from sandbox.verifier import verify` and `from agents.nodes
import ... assess_risk, generate_probe` — those bind local names in
agents.graph's own namespace, so THOSE call sites need agents.graph.verify
/ agents.graph.assess_risk / agents.graph.generate_probe patched
separately.
"""
from unittest.mock import patch
from types import SimpleNamespace

from agents.graph import modernize


class ScriptedLLM:
    """Returns responses from a fixed list, in order, clamping to the
    last entry once exhausted — so a single-item list means "always
    return this," and a longer list scripts an exact call sequence."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return SimpleNamespace(content=self.responses[idx])

    @property
    def call_count(self) -> int:
        return len(self.calls)


class QueuedVerify:
    """Returns results from a fixed list, in order, clamping to the last
    entry once exhausted. Records every candidate source it was called
    with, for assertions."""

    def __init__(self, results: list[dict]):
        self.results = results
        self.calls: list[str] = []

    def __call__(self, source_code, filename, run_cmd, **kwargs):
        self.calls.append(source_code)
        idx = min(len(self.calls) - 1, len(self.results) - 1)
        return self.results[idx]

    @property
    def call_count(self) -> int:
        return len(self.calls)


_OK = {"status": "success", "stderr": "", "stdout": "ok\n", "exit_code": 0}
_FAIL = {"status": "failed", "stderr": "compile error: bad syntax", "stdout": "", "exit_code": 1}
_RISK_NO = "RISK: no\nPure function, no side effects."

_CPP_SOURCE = b"int add(int a, int b) { return a + b; }\n"


def test_happy_path_succeeds_on_first_attempt():
    fake_llm = ScriptedLLM([
        "int add(int a, int b) { return a + b; }",  # refactor attempt 1
        _RISK_NO,                                     # risk assessment
    ])
    fake_verify = QueuedVerify([_OK])  # baseline + verifier both "ok"

    with patch("agents.nodes.llm", fake_llm), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify), \
         patch("agents.nodes.escalation_llm", None):
        final_state = modernize("cpp", _CPP_SOURCE, 0, len(_CPP_SOURCE.rstrip(b"\n")), max_iterations=5)

    assert final_state["status"] == "success"
    assert final_state["iteration_count"] == 1
    assert final_state["risk_flag"] is False


def test_retry_then_succeed_carries_error_into_second_attempt():
    fake_llm = ScriptedLLM([
        "int add(int a, int b) { return a - b; }",  # attempt 1 (verify will fail it)
        "int add(int a, int b) { return a + b; }",  # attempt 2 (verify will pass it)
        _RISK_NO,
    ])
    fake_verify = QueuedVerify([_OK, _FAIL, _OK])  # baseline, attempt1 fail, attempt2 ok

    with patch("agents.nodes.llm", fake_llm), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify), \
         patch("agents.nodes.escalation_llm", None):
        final_state = modernize("cpp", _CPP_SOURCE, 0, len(_CPP_SOURCE.rstrip(b"\n")), max_iterations=5)

    assert final_state["status"] == "success"
    assert final_state["iteration_count"] == 2
    # The second refactor call must have seen the first attempt's error —
    # this is the actual "does retry feed context back in" proof.
    second_call_text = str(fake_llm.calls[1])
    assert "compile error" in second_call_text


def test_gives_up_after_max_iterations():
    fake_llm = ScriptedLLM(["int add(int a, int b) { return a - b; }"])  # always wrong
    fake_verify = QueuedVerify([_OK, _FAIL])  # baseline ok, every verifier attempt fails

    with patch("agents.nodes.llm", fake_llm), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify), \
         patch("agents.nodes.escalation_llm", None):
        final_state = modernize("cpp", _CPP_SOURCE, 0, len(_CPP_SOURCE.rstrip(b"\n")), max_iterations=3)

    assert final_state["status"] == "gave_up"
    assert final_state["iteration_count"] == 3
    # gave_up means the fallback path ran, not a fluke of the loop
    # bottoming out some other way — no risk assessment should have run
    # since the chunk never succeeded.
    assert final_state["risk_flag"] is False


def test_escalates_to_stronger_model_after_threshold():
    base_llm = ScriptedLLM(["int add(int a, int b) { return a - b; }"])  # always wrong
    escalation_llm = ScriptedLLM(["int add(int a, int b) { return a + b; }"])  # gets it right
    fake_verify = QueuedVerify([_OK, _FAIL, _FAIL, _OK])  # baseline, 2 base failures, escalation success

    with patch("agents.nodes.llm", base_llm), \
         patch("agents.nodes.escalation_llm", escalation_llm), \
         patch("agents.nodes.ESCALATION_MODEL", "fake-strong-model"), \
         patch("agents.nodes.ESCALATION_THRESHOLD", 2), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify):
        final_state = modernize("cpp", _CPP_SOURCE, 0, len(_CPP_SOURCE.rstrip(b"\n")), max_iterations=5)

    assert final_state["status"] == "success"
    # base model tried exactly twice (iteration_count 0 and 1, both below
    # threshold 2), then escalation model took over and got it right on
    # its first attempt. base_llm gets ONE more call after that: risk
    # assessment always uses the base model by design, regardless of
    # escalation state (see assess_risk's docstring) — so 3 total, not 2.
    assert base_llm.call_count == 3
    assert escalation_llm.call_count == 1


def test_risk_assessment_result_flows_into_final_state():
    fake_llm = ScriptedLLM([
        "int add(int a, int b) { return a + b; }",
        "RISK: yes\nThis touches a global counter.",
    ])
    fake_verify = QueuedVerify([_OK])

    with patch("agents.nodes.llm", fake_llm), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify), \
         patch("agents.nodes.escalation_llm", None):
        final_state = modernize("cpp", _CPP_SOURCE, 0, len(_CPP_SOURCE.rstrip(b"\n")), max_iterations=5)

    assert final_state["status"] == "success"
    assert final_state["risk_flag"] is True
    assert "global counter" in final_state["risk_reason"]


def test_probe_mismatch_fails_even_when_whole_file_check_passes():
    # This is the actual value proposition of the probe feature: a
    # function the file's own entry point never calls gets zero coverage
    # from the whole-file baseline check. Only the probe catches it.
    python_source = b"def add(a, b):\n    return a + b\n"

    fake_llm = ScriptedLLM([
        "def add(a, b):\n    return a - b",  # WRONG — subtracts instead
    ])

    def fake_generate_probe(language, function_code):
        return "print(add(2, 3))"

    # Sequence inside modernize(): baseline call, probe-baseline call,
    # then verifier_node's whole-file call + probe call (repeated per
    # iteration since it never succeeds).
    fake_verify = QueuedVerify([
        _OK,                                                     # whole-file baseline
        {"status": "success", "stderr": "", "stdout": "5\n", "exit_code": 0},  # probe baseline: 2+3=5
        _OK,                                                     # attempt1 whole-file check (passes — file still runs)
        {"status": "success", "stderr": "", "stdout": "-1\n", "exit_code": 0},  # attempt1 probe: 2-3=-1, MISMATCH
    ])

    with patch("agents.nodes.llm", fake_llm), \
         patch("agents.nodes.verify", fake_verify), \
         patch("agents.graph.verify", fake_verify), \
         patch("agents.graph.generate_probe", fake_generate_probe), \
         patch("agents.nodes.escalation_llm", None):
        final_state = modernize("python", python_source, 0, len(python_source.rstrip(b"\n")), max_iterations=1)

    assert final_state["status"] == "gave_up"
    assert "different result" in final_state["compiler_stderr"]
