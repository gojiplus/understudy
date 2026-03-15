"""LangGraph adapter: wraps LangGraph agents for use with understudy."""

import logging
import uuid
from typing import Any

from ..mocks import MockToolkit
from ..runner import AgentApp, AgentResponse
from ..trace import ToolCall

logger = logging.getLogger(__name__)


class LangGraphApp(AgentApp):
    """Wraps a LangGraph CompiledGraph for use with understudy.

    Usage:
        from langgraph.graph import StateGraph
        from understudy.langgraph import LangGraphApp

        # Build your LangGraph agent
        graph = StateGraph(...)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        compiled = graph.compile()

        # Test with understudy
        app = LangGraphApp(graph=compiled)
        trace = run(app, scene)
        assert trace.called("search_orders")
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        config: dict[str, Any] | None = None,
    ):
        """
        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: The key used for message input in the graph state.
            config: Optional config dict passed to graph.stream().
        """
        self.graph = graph
        self.input_key = input_key
        self.config = config or {}
        self.thread_id: str | None = None
        self._mocks: MockToolkit | None = None

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the LangGraph session."""
        try:
            from langchain_core.messages import HumanMessage  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "langchain-core package required. Install with: pip install understudy[langgraph]"
            ) from e

        self.thread_id = str(uuid.uuid4())
        self._mocks = mocks
        logger.debug("Starting LangGraph session %s", self.thread_id)

    def send(self, message: str) -> AgentResponse:
        """Send a user message to the LangGraph agent and capture the response."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        except ImportError as e:
            raise ImportError(
                "langchain-core package required. Install with: pip install understudy[langgraph]"
            ) from e

        if self.thread_id is None:
            raise RuntimeError("LangGraphApp not started. Call start() first.")

        logger.debug("Sending message: %s", message[:100])

        input_state = {self.input_key: [HumanMessage(content=message)]}

        config = {**self.config}
        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"]["thread_id"] = self.thread_id

        tool_calls: list[ToolCall] = []
        tool_call_map: dict[str, ToolCall] = {}
        final_content = ""
        terminal_state: str | None = None
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0

        for chunk in self.graph.stream(input_state, config=config, stream_mode="updates"):
            for _node_name, updates in chunk.items():
                messages = updates.get("messages", [])
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                call = ToolCall(
                                    tool_name=tc["name"],
                                    arguments=tc.get("args", {}),
                                )
                                tool_calls.append(call)
                                tool_call_map[tc["id"]] = call
                                logger.debug("Tool call: %s", tc["name"])
                        if msg.content and isinstance(msg.content, str):
                            final_content = msg.content

                        # capture token usage from response_metadata
                        if hasattr(msg, "response_metadata") and msg.response_metadata:
                            usage = msg.response_metadata.get("token_usage", {})
                            input_tokens += usage.get("prompt_tokens", 0) or 0
                            output_tokens += usage.get("completion_tokens", 0) or 0
                            thinking_tokens += usage.get("reasoning_tokens", 0) or 0

                    elif isinstance(msg, ToolMessage):
                        tool_call_id = getattr(msg, "tool_call_id", None)
                        if tool_call_id and tool_call_id in tool_call_map:
                            tool_call_map[tool_call_id].result = msg.content
                        else:
                            tool_name = getattr(msg, "name", None)
                            if tool_name:
                                for call in reversed(tool_calls):
                                    if call.tool_name == tool_name and call.result is None:
                                        call.result = msg.content
                                        break

        # capture state snapshot from graph
        state_snapshot = None
        try:
            graph_state = self.graph.get_state(config)
            if graph_state and hasattr(graph_state, "values"):
                state_values = graph_state.values
                if isinstance(state_values, dict):
                    state_snapshot = {k: v for k, v in state_values.items() if k != "messages"}
        except Exception:
            pass

        return AgentResponse(
            content=final_content,
            tool_calls=tool_calls,
            terminal_state=terminal_state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            state_snapshot=state_snapshot,
        )

    def stop(self) -> None:
        """Clean up the LangGraph session."""
        self.thread_id = None
        self._mocks = None
        logger.debug("LangGraph session stopped")
