"""MockToolkit: mock tool handlers for testing agents."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Literal


class ToolError(Exception):
    """Raised by mock tools to signal an error to the agent."""


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

    def __init__(
        self,
        *,
        unmatched: Literal["allow", "error"] = "allow",
        state: dict[str, Any] | None = None,
    ):
        """Create handlers with an explicit policy for unmatched calls.

        Args:
            unmatched: Execute unregistered tools or reject them before execution.
            state: Initial mock-service state, copied for this toolkit.
        """
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._scoped: dict[tuple[tuple[str, ...], str], Callable[..., Any]] = {}
        self.unmatched = unmatched
        self.state = deepcopy(state or {})

    def handle(self, tool_name: str, *, agent_path: tuple[str, ...] = ()) -> Callable:
        """Register a handler, optionally scoped to an agent subtree.

        Args:
            tool_name: Tool to intercept.
            agent_path: Agent path prefix; the longest matching prefix wins.

        Returns:
            A handler registration decorator.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if agent_path:
                self._scoped[(agent_path, tool_name)] = fn
            else:
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
        """Names of all tools that have a registered mock handler."""
        return list(self._handlers.keys())

    def resolve(
        self, tool_name: str, agent_path: tuple[str, ...] = ()
    ) -> Callable[..., Any] | None:
        """Find a handler before executing a tool.

        Args:
            tool_name: Requested tool.
            agent_path: Invocation path for subtree selection.

        Returns:
            The handler, or None when real execution is allowed.

        Raises:
            ToolError: If an unmatched tool is forbidden.
        """
        for length in range(len(agent_path), 0, -1):
            handler = self._scoped.get((agent_path[:length], tool_name))
            if handler is not None:
                return handler
        handler = self._handlers.get(tool_name)
        if handler is None and self.unmatched == "error":
            raise ToolError(f"Unmocked tool: {tool_name} at {'/'.join(agent_path)}")
        return handler
