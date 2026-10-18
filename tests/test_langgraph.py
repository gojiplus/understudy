"""Tests for LangGraph adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from understudy import MockToolkit

pytest.importorskip("langgraph", reason="langgraph not installed")
pytest.importorskip("langchain_core", reason="langchain-core not installed")


@pytest.fixture(autouse=True)
def async_fake_streams(monkeypatch):
    from understudy.langgraph import LangGraphApp

    original = LangGraphApp.__init__

    def initialize(self, graph, *args, **kwargs):
        kwargs.setdefault("tool_interception", True)
        original(self, graph, *args, **kwargs)

        async def stream(*args, **kwargs):
            for item in graph.stream(*args, **kwargs):
                yield item

        graph.astream = stream

    monkeypatch.setattr(LangGraphApp, "__init__", initialize)


@pytest.mark.parametrize("state", [None, SimpleNamespace(values=[], next=("node",))])
def test_absent_or_non_mapping_state_and_orphan_result(state):
    from langchain_core.messages import ToolMessage

    from understudy.langgraph import LangGraphApp

    graph = Mock()
    graph.stream.return_value = [
        {
            "node": {
                "messages": [
                    ToolMessage(
                        content="orphan", tool_call_id="missing", name="unknown"
                    )
                ]
            }
        }
    ]
    graph.get_state.return_value = state
    graph.stream.return_value = (
        [((), c) for c in graph.stream.return_value]
        if isinstance(graph.stream.return_value, list)
        else graph.stream.return_value
    )
    app = LangGraphApp(graph)
    app.start()
    response = app.send("hello")
    assert response.state_snapshot is None
    assert response.terminal_state is None
    assert response.tool_calls == []
    app.stop()


def test_config_and_mock_scope():
    from langchain_core.messages import AIMessage

    from understudy.langgraph import LangGraphApp
    from understudy.langgraph.tools import (
        MockableToolContext,
        get_mock_toolkit,
        mockable_tool,
    )

    @mockable_tool
    def lookup():
        return "real"

    mocks = MockToolkit()
    mocks.handle("lookup")(lambda: "mocked")
    outer = MockToolkit()
    config = {"configurable": {"tenant": "customer"}}

    def stream(state, config, stream_mode, subgraphs):
        assert state["input"][0].content == "hello"
        assert config["configurable"]["tenant"] == "customer"
        assert config["configurable"]["thread_id"] == app.thread_id
        assert stream_mode == "updates"
        yield (), {"agent": {"messages": AIMessage(content=lookup())}}

    graph = Mock(
        stream=Mock(side_effect=stream),
        get_state=Mock(side_effect=ValueError("no checkpoint")),
    )
    graph.stream.return_value = (
        [((), c) for c in graph.stream.return_value]
        if isinstance(graph.stream.return_value, list)
        else graph.stream.return_value
    )
    app = LangGraphApp(graph, input_key="input", config=config)
    app.start(mocks)
    with MockableToolContext(outer):
        assert app.send("hello").content == "mocked"
        assert get_mock_toolkit() is outer
    assert get_mock_toolkit() is None
    assert config == {"configurable": {"tenant": "customer"}}
    app.stop()


@pytest.mark.parametrize("finished", [False, True])
def test_state_usage_and_unmatched_tool_results(finished):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from understudy.langgraph import LangGraphApp

    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="working",
            tool_calls=[
                {"name": "lookup", "args": {}, "id": None},
                {"name": "lookup", "args": {"id": 2}, "id": "known"},
            ],
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "reasoning_tokens": 1,
                }
            },
        ),
        ToolMessage(content="by-id", tool_call_id="known", name="lookup"),
        ToolMessage(content="by-name", tool_call_id="missing", name="lookup"),
        ToolMessage(content="ignored", tool_call_id="missing"),
        AIMessage(
            content="done",
            usage_metadata={"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
        ),
    ]
    graph = Mock()
    graph.stream.return_value = [{"node": {"messages": messages}}]
    graph.get_state.return_value = SimpleNamespace(
        values={"messages": messages, "order": "done"},
        next=() if finished else ("node",),
    )
    graph.stream.return_value = (
        [((), c) for c in graph.stream.return_value]
        if isinstance(graph.stream.return_value, list)
        else graph.stream.return_value
    )
    app = LangGraphApp(graph)
    app.start()
    result = app.send("hello")
    assert result.content == "done"
    assert [call.result for call in result.tool_calls] == ["by-name", "by-id"]
    assert result.state_snapshot == {"order": "done"}
    assert result.terminal_state is None
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == (
        10,
        5,
        1,
    )
    app.stop()


def test_mock_scope_restored_on_graph_error():
    from understudy.langgraph import LangGraphApp
    from understudy.langgraph.tools import MockableToolContext, get_mock_toolkit

    outer = MockToolkit()
    app = LangGraphApp(Mock(stream=Mock(side_effect=ValueError("graph failed"))))
    app.start(MockToolkit())
    with MockableToolContext(outer):
        with pytest.raises(ValueError, match="graph failed"):
            app.send("hello")
        assert get_mock_toolkit() is outer
    assert get_mock_toolkit() is None
    app.stop()


@pytest.mark.parametrize("operation", ["start", "send"])
def test_missing_langchain(monkeypatch, operation):
    import sys

    from understudy.langgraph import LangGraphApp

    monkeypatch.setitem(sys.modules, "langchain_core.messages", None)
    app = LangGraphApp(Mock())
    import asyncio

    action = (
        app.start
        if operation == "start"
        else lambda: asyncio.run(app._send_async("hello"))
    )
    with pytest.raises(ImportError, match=r"understudy\[langgraph\]"):
        action()


class TestLangGraphApp:
    def test_import(self):
        from understudy.langgraph import LangGraphApp

        assert LangGraphApp is not None

    def test_start_stop(self):
        from unittest.mock import MagicMock

        from understudy.langgraph import LangGraphApp

        mock_graph = MagicMock()
        app = LangGraphApp(graph=mock_graph)

        app.start()
        assert app.thread_id is not None

        app.stop()
        assert app.thread_id is None

    def test_send_extracts_tool_calls(self):
        from langchain_core.messages import AIMessage, ToolMessage

        from understudy.langgraph import LangGraphApp

        def mock_stream(input_state, config=None, stream_mode=None):
            yield {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "search_orders",
                                    "args": {"order_id": "ORD-123"},
                                    "id": "call_abc",
                                }
                            ],
                        )
                    ]
                }
            }
            yield {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content='{"status": "delivered"}',
                            tool_call_id="call_abc",
                            name="search_orders",
                        )
                    ]
                }
            }
            yield {
                "agent": {
                    "messages": [
                        AIMessage(content="Your order ORD-123 has been delivered.")
                    ]
                }
            }

        class MockGraph:
            def stream(
                self, input_state, config=None, stream_mode=None, subgraphs=False
            ):
                return (
                    ((), chunk)
                    for chunk in mock_stream(input_state, config, stream_mode)
                )

        app = LangGraphApp(graph=MockGraph())
        app.start()

        response = app.send("Where is my order ORD-123?")

        assert response.content == "Your order ORD-123 has been delivered."
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == "search_orders"
        assert response.tool_calls[0].arguments == {"order_id": "ORD-123"}
        assert response.tool_calls[0].result == '{"status": "delivered"}'

        app.stop()

    def test_send_multiple_tool_calls(self):
        from langchain_core.messages import AIMessage, ToolMessage

        from understudy.langgraph import LangGraphApp

        def mock_stream(input_state, config=None, stream_mode=None):
            yield {
                "agent": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "lookup_customer",
                                    "args": {"email": "test@example.com"},
                                    "id": "call_1",
                                },
                                {
                                    "name": "search_orders",
                                    "args": {"customer_id": "CUST-1"},
                                    "id": "call_2",
                                },
                            ],
                        )
                    ]
                }
            }
            yield {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content='{"customer_id": "CUST-1"}',
                            tool_call_id="call_1",
                            name="lookup_customer",
                        ),
                        ToolMessage(
                            content='[{"order_id": "ORD-1"}]',
                            tool_call_id="call_2",
                            name="search_orders",
                        ),
                    ]
                }
            }
            yield {
                "agent": {
                    "messages": [AIMessage(content="Found 1 order for customer.")]
                }
            }

        class MockGraph:
            def stream(
                self, input_state, config=None, stream_mode=None, subgraphs=False
            ):
                return (
                    ((), chunk)
                    for chunk in mock_stream(input_state, config, stream_mode)
                )

        app = LangGraphApp(graph=MockGraph())
        app.start()

        response = app.send("Find orders for test@example.com")

        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].tool_name == "lookup_customer"
        assert response.tool_calls[0].result == '{"customer_id": "CUST-1"}'
        assert response.tool_calls[1].tool_name == "search_orders"
        assert response.tool_calls[1].result == '[{"order_id": "ORD-1"}]'

        app.stop()

    def test_send_no_tool_calls(self):
        from langchain_core.messages import AIMessage

        from understudy.langgraph import LangGraphApp

        def mock_stream(input_state, config=None, stream_mode=None):
            yield {"agent": {"messages": [AIMessage(content="Hello! How can I help?")]}}

        class MockGraph:
            def stream(
                self, input_state, config=None, stream_mode=None, subgraphs=False
            ):
                return (
                    ((), chunk)
                    for chunk in mock_stream(input_state, config, stream_mode)
                )

        app = LangGraphApp(graph=MockGraph())
        app.start()

        response = app.send("Hi")

        assert response.content == "Hello! How can I help?"
        assert len(response.tool_calls) == 0

        app.stop()

    def test_not_started_raises(self):
        from unittest.mock import MagicMock

        from understudy.langgraph import LangGraphApp

        mock_graph = MagicMock()
        app = LangGraphApp(graph=mock_graph)

        with pytest.raises(RuntimeError, match="not started"):
            app.send("hello")


class TestMockableTools:
    def test_mockable_tool_decorator(self):
        from understudy.langgraph.tools import mockable_tool, set_mock_toolkit

        @mockable_tool
        def search_orders(order_id: str) -> dict:
            return {"order_id": order_id, "source": "real"}

        result = search_orders(order_id="ORD-1")
        assert result["source"] == "real"

        mocks = MockToolkit()

        @mocks.handle("search_orders")
        def mock_search(order_id: str) -> dict:
            return {"order_id": order_id, "source": "mock"}

        set_mock_toolkit(mocks)
        try:
            result = search_orders(order_id="ORD-1")
            assert result["source"] == "mock"
        finally:
            set_mock_toolkit(None)

        result = search_orders(order_id="ORD-1")
        assert result["source"] == "real"

    def test_mockable_tool_context(self):
        from understudy.langgraph.tools import MockableToolContext, mockable_tool

        @mockable_tool
        def get_status() -> str:
            return "real"

        mocks = MockToolkit()

        @mocks.handle("get_status")
        def mock_status() -> str:
            return "mock"

        assert get_status() == "real"

        with MockableToolContext(mocks):
            assert get_status() == "mock"

        assert get_status() == "real"

    def test_mockable_tool_no_handler(self):
        from understudy.langgraph.tools import mockable_tool, set_mock_toolkit

        @mockable_tool
        def unmocked_tool() -> str:
            return "real"

        mocks = MockToolkit()

        set_mock_toolkit(mocks)
        try:
            result = unmocked_tool()
            assert result == "real"
        finally:
            set_mock_toolkit(None)
