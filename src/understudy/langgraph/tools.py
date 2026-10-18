"""Tool interception scoped to each execution, including framework worker threads."""

import functools
import inspect
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any

from ..mocks import MockToolkit

_toolkit: ContextVar[MockToolkit | None] = ContextVar("understudy_tools", default=None)


def set_mock_toolkit(mocks: MockToolkit | None) -> None:
    """Set mocks in the current execution context."""
    _toolkit.set(mocks)


def get_mock_toolkit() -> MockToolkit | None:
    """Read the current execution's mocks."""
    return _toolkit.get()


def tool_path() -> tuple[str, ...]:
    """Read the native graph namespace when invoked inside a graph node."""
    try:
        from langgraph.config import get_config

        metadata = get_config().get("metadata", {})
    except RuntimeError:
        return ()
    namespace = metadata.get("langgraph_checkpoint_ns", "")
    path = tuple(part.split(":")[0] for part in namespace.split("|") if part)
    node = metadata.get("langgraph_node")
    return (*path, node) if node and (not path or path[-1] != node) else path


def mockable_tool[F: Callable[..., Any]](func: F) -> F:
    """Intercept a synchronous or asynchronous tool before executing it."""

    def handler():
        mocks = get_mock_toolkit()
        return mocks.resolve(func.__name__, tool_path()) if mocks is not None else None

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            result = (handler() or func)(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return (handler() or func)(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


class MockableToolContext:
    """Install mocks for one invocation and restore its caller's context."""

    def __init__(self, mocks: MockToolkit | None):
        """Configure mocks for this scope."""
        self.mocks = mocks
        self._token: Token | None = None

    def __enter__(self) -> "MockableToolContext":
        """Install mocks in the current context."""
        self._token = _toolkit.set(self.mocks)
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Restore context even when a tool raises."""
        if self._token is not None:
            _toolkit.reset(self._token)
