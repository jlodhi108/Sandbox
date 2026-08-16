from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.nodes import refactorer_node, verifier_node, fallback_node


def router(state: AgentState) -> str:
    if state["status"] == "success":
        return "success"
    if state["iteration_count"] >= state["max_iterations"]:
        return "give_up"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("refactorer", refactorer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("refactorer")
    graph.add_edge("refactorer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        router,
        {
            "success": END,
            "retry": "refactorer",
            "give_up": "fallback",
        },
    )
    graph.add_edge("fallback", END)

    return graph.compile()


def modernize(
    language: str,
    full_source: bytes,
    chunk_start: int,
    chunk_end: int,
    max_iterations: int = 5,
) -> AgentState:
    app = build_graph()
    original_code = full_source[chunk_start:chunk_end].decode("utf-8")
    initial_state: AgentState = {
        "language": language,
        "full_source": full_source,
        "chunk_start": chunk_start,
        "chunk_end": chunk_end,
        "original_code": original_code,
        "modernized_code": "",
        "required_imports": [],
        "compiler_stderr": "",
        "iteration_count": 0,
        "status": "pending",
        "max_iterations": max_iterations,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    from languages import get_handler

    with open("legacy_samples/legacy.cpp", "rb") as f:
        source = f.read()

    # demo: modernize just the last chunk in isolation
    handler = get_handler("legacy_samples/legacy.cpp")
    chunks = handler.chunk(source)
    last = chunks[-1]

    final_state = modernize(handler.name, source, last.start_byte, last.end_byte)
    print("--- STATUS:", final_state["status"], "---")
    print("--- ITERATIONS:", final_state["iteration_count"], "---")
    print(final_state["modernized_code"])
