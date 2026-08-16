from agents.review_graph import start_review, resume_review

_NO_FLAGS = {
    "risk_flag": False, "risk_reason": "",
    "security_flag": False, "security_findings": [],
    "mutation_confidence_flag": False, "mutation_confidence_reason": "",
}


def test_start_review_auto_approves_when_no_flags_set():
    result = start_review(_NO_FLAGS)
    assert result["status"] == "approved"
    assert result["interrupt_value"] is None


def test_start_review_pauses_when_risk_flag_set():
    flags = {**_NO_FLAGS, "risk_flag": True, "risk_reason": "touches global state"}
    result = start_review(flags)
    assert result["status"] == "awaiting_review"
    assert result["interrupt_value"]["risk_flag"] is True
    assert result["interrupt_value"]["risk_reason"] == "touches global state"


def test_start_review_pauses_when_security_flag_set():
    flags = {**_NO_FLAGS, "security_flag": True, "security_findings": [{"rule_id": "x"}]}
    result = start_review(flags)
    assert result["status"] == "awaiting_review"
    assert result["interrupt_value"]["security_flag"] is True


def test_start_review_pauses_when_mutation_confidence_flag_set():
    flags = {**_NO_FLAGS, "mutation_confidence_flag": True, "mutation_confidence_reason": "weak coverage"}
    result = start_review(flags)
    assert result["status"] == "awaiting_review"
    assert result["interrupt_value"]["mutation_confidence_flag"] is True


def test_resume_review_approve():
    flags = {**_NO_FLAGS, "risk_flag": True, "risk_reason": "x"}
    started = start_review(flags)
    resumed = resume_review(started["thread_id"], approved=True)
    assert resumed["status"] == "approved"


def test_resume_review_reject():
    flags = {**_NO_FLAGS, "risk_flag": True, "risk_reason": "x"}
    started = start_review(flags)
    resumed = resume_review(started["thread_id"], approved=False)
    assert resumed["status"] == "rejected"


def test_each_review_gets_an_independent_thread_id():
    flags = {**_NO_FLAGS, "risk_flag": True, "risk_reason": "x"}
    a = start_review(flags)
    b = start_review(flags)
    assert a["thread_id"] != b["thread_id"]
    # Resolving one must not affect the other.
    resume_review(a["thread_id"], approved=True)
    resumed_b = resume_review(b["thread_id"], approved=False)
    assert resumed_b["status"] == "rejected"
