"""ADK app plugin for nested tracing and tool interception."""

import inspect

from google.adk.plugins.base_plugin import BasePlugin

from ..instrumentation import ExecutionRecorder
from ..mocks import MockToolkit


class TestPlugin(BasePlugin):
    """Observe the whole ADK app without replacing agent callbacks."""

    __test__ = False

    def __init__(self, recorder: ExecutionRecorder, mocks: MockToolkit | None):
        """Bind instrumentation to a session.

        Args:
            recorder: Invocation recorder.
            mocks: Tool replacements and unmatched-call policy.
        """
        super().__init__(name="understudy")
        self.recorder = recorder
        self.mocks = mocks

    def _scope(self, context):
        return self.recorder.observe(context.node_path, context.invocation_id)

    async def before_agent_callback(self, *, agent, callback_context):  # noqa: ARG002
        """Record native agent invocation identity."""
        self._scope(callback_context)
        return

    async def after_agent_callback(self, *, agent, callback_context):  # noqa: ARG002
        """Close this agent's execution."""
        self._scope(callback_context).status = "completed"
        return

    async def on_agent_error_callback(self, *, agent, callback_context, error):  # noqa: ARG002
        """Record failed invocations without swallowing the framework error."""
        scope = self._scope(callback_context)
        scope.status, scope.error = "error", str(error)
        return

    async def on_event_callback(self, *, invocation_context, event):  # noqa: ARG002
        """Observe native workflow paths, including non-LLM workflow parents."""
        if event.node_info.path:
            self.recorder.observe(event.node_info.path, event.invocation_id)
        return

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        """Replace tools before any real side effect can execute."""
        scope = self._scope(tool_context)
        if self.mocks is None:
            return None
        handler = self.mocks.resolve(tool.name, scope.agent_path)
        if handler is None:
            return None
        result = handler(**tool_args)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError("ADK mock tools must return a dictionary")
        return result
