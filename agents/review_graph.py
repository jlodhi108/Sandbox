"""A tiny, SEPARATE LangGraph graph whose only job is human-in-the-loop
review of a flagged chunk, using LangGraph's native interrupt()/Command
mechanism. Deliberately kept separate from the main modernize() graph
(agents/graph.py) rather than added as a node there: LangGraph
re-executes a node's ENTIRE body from the start on every resume (see
interrupt()'s own docstring), so bundling a review step into a node
that also runs assess_risk/scan_security/check_mutation_confidence
would re-run those expensive, LLM/Docker-driven, non-deterministic
checks on every resume — this graph's node does nothing but check
pre-computed flags and (maybe) pause, which is free to re-execute.

Uses an in-memory checkpointer (MemorySaver), so a paused review only
survives for the lifetime of THIS PROCESS. That's long enough to review
across separate MCP tool calls within one long-lived server session
(exactly the use case this targets — see mcp_server.py's
resume_chunk_review tool) or across a synchronous prompt within one CLI
run (see main.py's --interactive handling), but NOT across a server
restart or separate CLI invocations. That's a deliberate scope
boundary, not a bug: full cross-process persistence needs a
database-backed checkpointer (a new dependency,
langgraph-checkpoint-sqlite) that isn't justified for a first version
of this feature.
"""
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class ReviewState(TypedDict):
    risk_flag: bool
    risk_reason: str
    security_flag: bool
    security_findings: list
    mutation_confidence_flag: bool
    mutation_confidence_reason: str
    decision: str  # "" | "approve" | "reject"


def _review_node(state: ReviewState) -> dict:
    if not (state["risk_flag"] or state["security_flag"] or state["mutation_confidence_flag"]):
        return {"decision": "approve"}
    decision = interrupt({
        "risk_flag": state["risk_flag"],
        "risk_reason": state["risk_reason"],
        "security_flag": state["security_flag"],
        "security_findings": state["security_findings"],
        "mutation_confidence_flag": state["mutation_confidence_flag"],
        "mutation_confidence_reason": state["mutation_confidence_reason"],
    })
    return {"decision": "approve" if decision == "approve" else "reject"}


_builder = StateGraph(ReviewState)
_builder.add_node("review", _review_node)
_builder.add_edge(START, "review")
_checkpointer = MemorySaver()
_review_app = _builder.compile(checkpointer=_checkpointer)


def start_review(flags: dict) -> dict:
    """flags: {"risk_flag", "risk_reason", "security_flag",
    "security_findings", "mutation_confidence_flag",
    "mutation_confidence_reason"} — exactly the fields already present
    on a successful modernize() final_state. Returns {"thread_id",
    "status": "approved" | "awaiting_review", "interrupt_value":
    dict | None}. "approved" with no pause at all happens when none of
    the three flags are set — the review graph's node auto-approves
    without ever calling interrupt(), so the common (unflagged) case
    costs nothing beyond one cheap in-memory graph invocation."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = _review_app.invoke({**flags, "decision": ""}, config)
    if "__interrupt__" in result:
        return {
            "thread_id": thread_id,
            "status": "awaiting_review",
            "interrupt_value": result["__interrupt__"][0].value,
        }
    return {"thread_id": thread_id, "status": "approved", "interrupt_value": None}


def resume_review(thread_id: str, approved: bool) -> dict:
    """Continue a paused review started by start_review(). Returns
    {"status": "approved" | "rejected"}. Raises if thread_id doesn't
    correspond to a paused review IN THIS PROCESS (already resolved,
    never started, or the process that started it has since exited —
    see this module's docstring on the in-memory checkpointer's scope)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = _review_app.invoke(Command(resume="approve" if approved else "reject"), config)
    return {"status": "approved" if result.get("decision") == "approve" else "rejected"}
