"""Native LangChain middleware that intercepts synchronous and async tools."""

import inspect
import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..langgraph.tools import get_mock_toolkit, tool_path


class MockTools(AgentMiddleware):
    """Return configured tool responses before calling the real handler."""

    def wrap_tool_call(self, request, handler):
        """Intercept a synchronous tool invocation."""
        mocks = get_mock_toolkit()
        mock = (
            mocks.resolve(request.tool_call["name"], tool_path())
            if mocks is not None
            else None
        )
        if mock is None:
            return handler(request)
        result = mock(**request.tool_call["args"])
        return ToolMessage(
            content=json.dumps(result), tool_call_id=request.tool_call["id"]
        )

    async def awrap_tool_call(self, request, handler):
        """Intercept an asynchronous tool invocation."""
        mocks = get_mock_toolkit()
        mock = (
            mocks.resolve(request.tool_call["name"], tool_path())
            if mocks is not None
            else None
        )
        if mock is None:
            return await handler(request)
        result = mock(**request.tool_call["args"])
        if inspect.isawaitable(result):
            result = await result
        return ToolMessage(
            content=json.dumps(result), tool_call_id=request.tool_call["id"]
        )
