"""Native graph checkpoints, namespaces, and concurrent tool interception."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from understudy import Checkpoint, MockToolkit, StateFixture, run_turn
from understudy.langgraph import LangGraphApp
from understudy.langgraph.tools import mockable_tool


class State(TypedDict):
    messages: Annotated[list, add_messages]
    n: int


def counter(state):
    n = state.get("n", 0) + 1
    return {"n": n, "messages": [AIMessage(content=str(n))]}


def graph_app():
    graph = (
        StateGraph(State)
        .add_node("counter", counter)
        .add_edge(START, "counter")
        .add_edge("counter", END)
        .compile(checkpointer=InMemorySaver())
    )
    return LangGraphApp(graph, app_id="counter-v1")


def test_native_checkpoint_and_forks(tmp_path):
    first = run_turn(
        graph_app,
        "hello",
        fixture=StateFixture(state={"n": 4}),
        capture_checkpoint=True,
    )
    first.checkpoint.save(tmp_path / "graph.json")
    saved = Checkpoint.load(tmp_path / "graph.json")
    a = run_turn(graph_app, "next", checkpoint=saved)
    b = run_turn(graph_app, "next", checkpoint=saved, state_patch={"n": 20})
    assert a.state_before["n"] == 5
    assert a.state_after["n"] == 6
    assert b.state_after["n"] == 21
    assert len(a.state_after["messages"]) == 4
    assert len(a.trace.turns) == 2
    assert saved.state["n"] == 5
    assert a.trace.executions[-1].agent_name == "counter"


@pytest.mark.parametrize("persistent_child", [None, True])
def test_nested_graph_observations(tmp_path, persistent_child):
    def factory():
        child = (
            StateGraph(State)
            .add_node("specialist", counter)
            .add_edge(START, "specialist")
            .add_edge("specialist", END)
            .compile(checkpointer=persistent_child)
        )
        parent = (
            StateGraph(State)
            .add_node("team", child)
            .add_edge(START, "team")
            .add_edge("team", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(parent)

    result = run_turn(factory, "hello", capture_checkpoint=True)
    result.checkpoint.save(tmp_path / "nested.json")
    restored = run_turn(
        factory, "next", checkpoint=Checkpoint.load(tmp_path / "nested.json")
    )
    assert restored.state_after["n"] == 2
    assert len(restored.state_after["messages"]) == 4
    spans = result.trace.executions
    assert "specialist" in result.trace.agents_invoked()
    specialist = next(s for s in spans if s.agent_name == "specialist")
    assert specialist.parent_id is not None
    assert "team" in specialist.agent_path


def test_concurrent_mocks_are_scoped_to_native_worker_contexts():
    barrier = Barrier(2)
    handler_barrier = Barrier(2)
    real = []

    @mockable_tool
    def lookup():
        real.append(True)
        return "real"

    def node(state):
        barrier.wait(timeout=5)
        return {"messages": [AIMessage(content=lookup())]}

    def app():
        graph = (
            StateGraph(State)
            .add_node("lookup", node)
            .add_edge(START, "lookup")
            .add_edge("lookup", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph, tool_interception=True)

    def execute(value):
        def mocks():
            toolkit = MockToolkit(unmatched="error")

            def mocked():
                handler_barrier.wait(timeout=5)
                return value

            toolkit.handle("lookup")(mocked)
            return toolkit

        return run_turn(app, "hello", mocks_factory=mocks).trace.turns[-1].content

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(execute, ["a", "b"])) == ["a", "b"]
    assert real == []


def test_pending_checkpoint_rejected_and_async_tools():
    from langgraph.types import interrupt

    from understudy import UnsupportedCapabilityError

    async def node(state):
        interrupt("approve?")
        return {"n": 1}

    def app():
        graph = (
            StateGraph(State)
            .add_node("approval", node)
            .add_edge(START, "approval")
            .add_edge("approval", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph)

    result = run_turn(app, "go")
    assert result.trace.terminal_state == "interrupted"
    with pytest.raises(UnsupportedCapabilityError, match="pending"):
        run_turn(app, "go", capture_checkpoint=True)


def test_async_decorated_tool():
    from understudy.langgraph.tools import MockableToolContext

    @mockable_tool
    async def lookup(value):
        return "real " + value

    toolkit = MockToolkit()
    toolkit.handle("lookup")(lambda value: "mock " + value)

    async def invoke():
        with MockableToolContext(toolkit):
            assert await lookup("a") == "mock a"
        assert await lookup("a") == "real a"

    import asyncio

    asyncio.run(invoke())


def test_synthetic_tool_history_and_version_rejection():
    from understudy import ToolCall, Turn

    fixture = StateFixture(
        state={"n": 4},
        history=[
            Turn(role="user", content="old"),
            Turn(
                role="agent",
                content="old answer",
                tool_calls=[
                    ToolCall(tool_name="old", arguments={"x": 1}, result={"ok": True})
                ],
            ),
        ],
    )
    result = run_turn(graph_app, "next", fixture=fixture, capture_checkpoint=True)
    assert len(result.state_after["messages"]) == 5
    assert not result.trace.called("old")
    result.checkpoint.framework_version = "different"
    with pytest.raises(ValueError, match="framework version"):
        run_turn(graph_app, "next", checkpoint=result.checkpoint)


def test_nested_agents_with_identical_tool_call_ids():
    from langchain_core.messages import ToolMessage

    class Messages(TypedDict):
        messages: Annotated[list, add_messages]

    def child(label):
        def model(state):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"id": "same", "name": "lookup", "args": {}}],
                    )
                ]
            }

        def tools(state):
            return {
                "messages": [
                    ToolMessage(content=label, name="lookup", tool_call_id="same")
                ]
            }

        return (
            StateGraph(Messages)
            .add_node("model", model)
            .add_node("tools", tools)
            .add_edge(START, "model")
            .add_edge("model", "tools")
            .add_edge("tools", END)
            .compile()
        )

    def factory():
        graph = (
            StateGraph(Messages)
            .add_node("a", child("a"))
            .add_node("b", child("b"))
            .add_edge(START, "a")
            .add_edge(START, "b")
            .add_edge("a", END)
            .add_edge("b", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph)

    result = run_turn(factory, "go")
    assert len(result.trace.tool_calls) == 2
    for call in result.trace.tool_calls:
        assert call.result in call.agent_path
