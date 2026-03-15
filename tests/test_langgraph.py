"""Tests for LangGraph adapter."""

import pytest

from understudy import MockToolkit

pytest.importorskip("langgraph", reason="langgraph not installed")
pytest.importorskip("langchain_core", reason="langchain-core not installed")


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
                "agent": {"messages": [AIMessage(content="Your order ORD-123 has been delivered.")]}
            }

        class MockGraph:
            def stream(self, input_state, config=None, stream_mode=None):
                return mock_stream(input_state, config, stream_mode)

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
            yield {"agent": {"messages": [AIMessage(content="Found 1 order for customer.")]}}

        class MockGraph:
            def stream(self, input_state, config=None, stream_mode=None):
                return mock_stream(input_state, config, stream_mode)

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
            def stream(self, input_state, config=None, stream_mode=None):
                return mock_stream(input_state, config, stream_mode)

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
