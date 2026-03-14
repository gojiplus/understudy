"""MockToolkit: mock tool handlers for testing agents."""

from collections.abc import Callable
from typing import Any


class ToolError(Exception):
    """Raised by mock tools to signal an error to the agent."""

    pass


class MockToolkit:
    """A collection of mock tool handlers for testing.

    Usage::

        mocks = MockToolkit()

        @mocks.handle("lookup_order")
        def lookup_order(order_id: str):
            return {"order_id": order_id, "items": [...]}

        @mocks.handle("create_return")
        def create_return(order_id: str, item_sku: str, reason: str):
            return {"return_id": "RET-001", "status": "created"}

        trace = run(app, scene, mocks=mocks)
    """

    def __init__(self):
        self._handlers: dict[str, Callable[..., Any]] = {}

    def handle(self, tool_name: str) -> Callable:
        """Decorator to register a custom mock handler for a tool."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._handlers[tool_name] = fn
            return fn

        return decorator

    def get_handler(self, tool_name: str) -> Callable[..., Any] | None:
        """Get the mock handler for a tool, or None if not mocked."""
        return self._handlers.get(tool_name)

    def call(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a mock tool. Raises KeyError if no handler registered."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            raise KeyError(f"No mock handler for tool '{tool_name}'")
        return handler(**kwargs)

    @property
    def available_tools(self) -> list[str]:
        return list(self._handlers.keys())
