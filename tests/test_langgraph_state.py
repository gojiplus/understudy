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


def test_private_child_history_is_excluded_after_restore():
    from langchain_core.messages import ToolMessage

    def factory():
        def child_node(state):
            n = state.get("n", 0) + 1
            new = (
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "only_first", "args": {}, "id": "first"}],
                    ),
                    ToolMessage(content="first result", tool_call_id="first"),
                ]
                if n == 1
                else [AIMessage(content="no new tools")]
            )
            return {"n": n, "messages": state["messages"] + new}

        child = (
            StateGraph(State)
            .add_node("child_model", child_node)
            .add_edge(START, "child_model")
            .add_edge("child_model", END)
            .compile(checkpointer=True)
        )

        def wrapper(state):
            child.invoke({"messages": [state["messages"][-1]]})
            return {"messages": [AIMessage(content="root done")]}

        graph = (
            StateGraph(State)
            .add_node("wrapper", wrapper)
            .add_edge(START, "wrapper")
            .add_edge("wrapper", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph)

    first = run_turn(factory, "one", capture_checkpoint=True)
    assert first.trace.called("only_first")
    second = run_turn(factory, "two", checkpoint=first.checkpoint)
    assert not second.trace.called("only_first")


def test_sibling_message_ids_are_independent():
    from langchain_core.messages import ToolMessage

    def child(label):
        def node(state):
            return {
                "messages": [
                    AIMessage(
                        id="same-ai",
                        content="",
                        tool_calls=[{"id": "call", "name": "lookup", "args": {}}],
                    ),
                    ToolMessage(id="same-tool", content=label, tool_call_id="call"),
                ]
            }

        return (
            StateGraph(State)
            .add_node("model", node)
            .add_edge(START, "model")
            .add_edge("model", END)
            .compile(checkpointer=True)
        )

    def factory():
        a, b = child("a"), child("b")

        def call_a(state):
            a.invoke({"messages": [state["messages"][-1]]})
            return {"messages": [AIMessage(content="a done")]}

        def call_b(state):
            b.invoke({"messages": [state["messages"][-1]]})
            return {"messages": [AIMessage(content="b done")]}

        graph = (
            StateGraph(State)
            .add_node("a", call_a)
            .add_node("b", call_b)
            .add_edge(START, "a")
            .add_edge(START, "b")
            .add_edge("a", END)
            .add_edge("b", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph)

    result = run_turn(factory, "go")
    assert sorted(c.result for c in result.trace.tool_calls) == ["a", "b"]


def test_binary_runtime_state_is_rejected_at_checkpoint_capture():
    class BinaryState(State):
        data: bytes

    def factory():
        def node(state):
            return {"messages": [AIMessage(content="ok")], "data": b"\xff\x00"}

        return LangGraphApp(
            StateGraph(BinaryState)
            .add_node("worker", node)
            .add_edge(START, "worker")
            .add_edge("worker", END)
            .compile(checkpointer=InMemorySaver())
        )

    assert run_turn(factory, "go").state_after["data"] == b"\xff\x00"
    with pytest.raises(ValueError, match="JSON-compatible"):
        run_turn(factory, "go", capture_checkpoint=True)


def test_checkpoint_export_work_grows_linearly():
    class CountingSerde:
        def __init__(self, wrapped):
            self.wrapped, self.calls, self.bytes = wrapped, 0, 0

        def loads_typed(self, data):
            self.calls += 1
            self.bytes += len(data[1])
            return self.wrapped.loads_typed(data)

        def dumps_typed(self, value):
            return self.wrapped.dumps_typed(value)

    work = []
    for turns in [10, 40]:
        app = graph_app()
        saver = app.graph.checkpointer
        serde = CountingSerde(saver.serde)
        saver.serde = serde
        app.start()
        try:
            for _ in range(turns):
                app.send("next")
            serde.calls = serde.bytes = 0
            checkpoint = app.checkpoint()
            assert checkpoint.state["n"] == turns
            work.append((serde.calls, serde.bytes))
        finally:
            app.stop()
    assert work[1][0] <= 2 * work[0][0]
    assert work[1][1] <= 6 * work[0][1]


def test_fresh_child_invocations_can_reuse_message_ids():
    from langchain_core.messages import ToolMessage

    def factory():
        def child_node(state):
            return {
                "messages": [
                    AIMessage(
                        id="first-message",
                        content="",
                        tool_calls=[{"name": "lookup", "id": "call", "args": {}}],
                    ),
                    ToolMessage(content="new result", tool_call_id="call"),
                ]
            }

        child = (
            StateGraph(State)
            .add_node("model", child_node)
            .add_edge(START, "model")
            .add_edge("model", END)
            .compile()
        )

        def wrapper(state):
            child.invoke({"messages": [state["messages"][-1]]})
            return {"messages": [AIMessage(content="root done")]}

        graph = (
            StateGraph(State)
            .add_node("wrapper", wrapper)
            .add_edge(START, "wrapper")
            .add_edge("wrapper", END)
            .compile(checkpointer=InMemorySaver())
        )
        return LangGraphApp(graph)

    first = run_turn(factory, "one", capture_checkpoint=True)
    second = run_turn(factory, "two", checkpoint=first.checkpoint)
    assert len(first.trace.tool_calls) == len(second.trace.tool_calls) == 1
