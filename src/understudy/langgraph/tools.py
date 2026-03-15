"""Mockable tool wrapper for LangGraph tools."""

import functools
from collections.abc import Callable
from typing import Any

from ..mocks import MockToolkit

_global_mock_toolkit: MockToolkit | None = None


def set_mock_toolkit(mocks: MockToolkit | None) -> None:
    """Set the active MockToolkit globally.

    This should be called before invoking the LangGraph graph
    to enable mock responses for tools decorated with @mockable_tool.

    Note: Uses a global variable to support LangGraph's thread-pooled
    tool execution. For concurrent test execution, ensure tests are
    isolated or use separate processes.

    Usage:
        from understudy.langgraph.tools import set_mock_toolkit

        mocks = MockToolkit()
        @mocks.handle("search_orders")
        def mock_search(order_id: str):
            return {"order_id": order_id, "status": "delivered"}

        set_mock_toolkit(mocks)
        try:
            result = compiled_graph.invoke(...)
        finally:
            set_mock_toolkit(None)
    """
    global _global_mock_toolkit
    _global_mock_toolkit = mocks


def get_mock_toolkit() -> MockToolkit | None:
    """Get the active MockToolkit."""
    return _global_mock_toolkit


def mockable_tool[F: Callable[..., Any]](func: F) -> F:
    """Decorator that makes a LangGraph tool mockable during testing.

    When a MockToolkit is active (via set_mock_toolkit), the decorator
    checks if there's a mock handler registered for this tool. If so,
    it returns the mock response instead of executing the real function.

    Usage:
        from understudy.langgraph.tools import mockable_tool

        @mockable_tool
        def search_orders(order_id: str) -> dict:
            # Real implementation
            return db.search(order_id)

        # In production: executes real implementation
        # During testing with set_mock_toolkit: returns mock response
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        mocks = get_mock_toolkit()
        if mocks is not None:
            tool_name = func.__name__
            handler = mocks.get_handler(tool_name)
            if handler is not None:
                return handler(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class MockableToolContext:
    """Context manager for setting up mock toolkit.

    Usage:
        mocks = MockToolkit()
        @mocks.handle("search_orders")
        def mock_search(order_id: str):
            return {"found": True}

        with MockableToolContext(mocks):
            result = compiled_graph.invoke(...)
    """

    def __init__(self, mocks: MockToolkit | None):
        self.mocks = mocks
        self._previous: MockToolkit | None = None

    def __enter__(self) -> "MockableToolContext":
        self._previous = get_mock_toolkit()
        set_mock_toolkit(self.mocks)
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        set_mock_toolkit(self._previous)
