"""Measure adapter and checkpoint overhead without model or tool API calls."""

import asyncio
import json
import sys
from statistics import median
from time import perf_counter
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from understudy import Checkpoint
from understudy.langgraph import LangGraphApp


class State(TypedDict):
    """Conversation retained by the native graph."""

    messages: Annotated[list, add_messages]


def respond(state: State) -> dict:  # noqa: ARG001
    """Return deterministic text without calling a model."""
    return {"messages": [AIMessage(content="response" * 20)]}


def build_app() -> LangGraphApp:
    """Create a fresh graph and checkpoint store for each measurement."""
    graph = (
        StateGraph(State)
        .add_node("agent", respond)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile(checkpointer=InMemorySaver())
    )
    return LangGraphApp(graph)


def measure(turns: int) -> dict[str, float]:
    """Measure native turns, instrumented turns, and a portable checkpoint fork."""
    native = build_app()
    with asyncio.Runner() as loop:
        started = perf_counter()
        for _ in range(turns):
            loop.run(
                native.graph.ainvoke(
                    {"messages": [HumanMessage(content="prompt" * 20)]},
                    config={"configurable": {"thread_id": "baseline"}},
                )
            )
        native_ms = (perf_counter() - started) * 1000 / turns

    app = build_app()
    app.start()
    try:
        started = perf_counter()
        for _ in range(turns):
            app.send("prompt" * 20)
        adapter_ms = (perf_counter() - started) * 1000 / turns
        started = perf_counter()
        checkpoint = app.checkpoint()
        capture_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        encoded = checkpoint.model_dump_json()
        loaded = Checkpoint.model_validate_json(encoded)
        json_ms = (perf_counter() - started) * 1000
    finally:
        app.stop()

    restored = build_app()
    restored.start()
    try:
        started = perf_counter()
        restored.restore(loaded)
        restore_ms = (perf_counter() - started) * 1000
        started = perf_counter()
        restored.send("next")
        next_ms = (perf_counter() - started) * 1000
    finally:
        restored.stop()
    return {
        "native_ms_per_turn": native_ms,
        "adapter_ms_per_turn": adapter_ms,
        "capture_ms": capture_ms,
        "json_roundtrip_ms": json_ms,
        "restore_ms": restore_ms,
        "next_turn_ms": next_ms,
        "checkpoint_bytes": len(encoded.encode()),
    }


def main() -> None:
    """Print median measurements from three runs at each history size."""
    measure(5)
    results = []
    for turns in [50, 100, 200]:
        samples = [measure(turns) for _ in range(3)]
        results.append(
            {
                "turns": turns,
                **{
                    key: round(median(s[key] for s in samples), 3) for key in samples[0]
                },
            }
        )
    sys.stdout.write(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
