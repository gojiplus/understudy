"""ADK adapter: wraps Google ADK agents for use with understudy."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from ..mocks import MockToolkit
from ..runner import AgentApp, AgentResponse
from ..trace import AgentTransfer, ToolCall

logger = logging.getLogger(__name__)


def _load_dotenv():
    """Load environment variables from .env file if present."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


# Load .env on module import
_load_dotenv()


def _create_mock_callback(mocks: MockToolkit | None):
    """Create a before_tool_callback that returns mock responses.

    Args:
        mocks: MockToolkit instance or None.

    Returns:
        A callback function compatible with google-adk's before_tool_callback.
        Returns dict to bypass real tool execution (mock response).
        Returns None to allow normal execution.
    """

    def callback(tool, args: dict[str, Any], tool_context) -> dict | None:
        if mocks is None:
            return None
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        if mocks.get_handler(tool_name):
            try:
                result = mocks.call(tool_name, **args)
                return result
            except Exception as e:
                return {"error": str(e)}
        return None

    return callback


class ADKApp(AgentApp):
    """Wraps a Google ADK Agent for use with understudy.

    Usage:
        from google.adk import Agent
        from understudy.adk import ADKApp

        agent = Agent(model="gemini-2.5-flash", name="my_agent", ...)
        app = ADKApp(agent=agent)
        trace = run(app, scene)
    """

    def __init__(self, agent: Any, session_id: str | None = None):
        """
        Args:
            agent: A google.adk.Agent instance.
            session_id: Optional session ID. If None, a random one is generated.
        """
        self.agent = agent
        self.session_id = session_id
        self._runner = None
        self._session = None
        self._session_service = None
        self._mocks: MockToolkit | None = None
        self._current_agent: str | None = None
        self._agent_transfers: list[AgentTransfer] = []

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the ADK session."""
        asyncio.run(self._start_async(mocks))

    async def _start_async(self, mocks: MockToolkit | None = None) -> None:
        """Async implementation of start."""
        try:
            from google.adk import Runner
            from google.adk.sessions import InMemorySessionService
        except ImportError as e:
            raise ImportError(
                "google-adk package required. Install with: pip install understudy[adk]"
            ) from e
        import uuid

        # Suppress verbose Google SDK warnings
        logging.getLogger("google.adk").setLevel(logging.WARNING)
        logging.getLogger("google.genai").setLevel(logging.ERROR)

        self._mocks = mocks
        self._current_agent = getattr(self.agent, "name", None)
        self._agent_transfers = []
        self._session_id = self.session_id or str(uuid.uuid4())

        logger.debug("Starting ADK session %s for agent %s", self._session_id, self._current_agent)

        self._session_service = InMemorySessionService()
        if mocks:
            self.agent.before_tool_callback = _create_mock_callback(mocks)
            logger.debug("Registered %d mock handlers", len(mocks.available_tools))

        self._runner = Runner(
            agent=self.agent,
            app_name="understudy_test",
            session_service=self._session_service,
        )
        self._session = await self._session_service.create_session(
            app_name="understudy_test",
            user_id="understudy_user",
            session_id=self._session_id,
        )
        logger.debug("ADK session started")

    def send(self, message: str) -> AgentResponse:
        """Send a user message to the ADK agent and capture the response."""
        return asyncio.run(self._send_async(message))

    async def _send_async(self, message: str) -> AgentResponse:
        """Async implementation of send."""
        try:
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-adk package required. Install with: pip install understudy[adk]"
            ) from e

        logger.debug("Sending message: %s", message[:100])

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )

        tool_calls: list[ToolCall] = []
        agent_text_parts: list[str] = []
        terminal_state: str | None = None
        current_agent_name = self._current_agent
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0

        async for event in self._runner.run_async(
            user_id="understudy_user",
            session_id=self._session.id,
            new_message=user_content,
        ):
            # track agent attribution from event.author
            if hasattr(event, "author") and event.author:
                event_agent = event.author
                if event_agent != current_agent_name and current_agent_name:
                    self._agent_transfers.append(
                        AgentTransfer(
                            from_agent=current_agent_name,
                            to_agent=event_agent,
                            timestamp=datetime.now(UTC),
                        )
                    )
                current_agent_name = event_agent

            # detect explicit transfer_to_agent actions
            if (
                hasattr(event, "actions")
                and event.actions
                and hasattr(event.actions, "transfer_to_agent")
                and event.actions.transfer_to_agent
            ):
                target_agent = event.actions.transfer_to_agent
                if current_agent_name and target_agent != current_agent_name:
                    self._agent_transfers.append(
                        AgentTransfer(
                            from_agent=current_agent_name,
                            to_agent=target_agent,
                            timestamp=datetime.now(UTC),
                        )
                    )
                current_agent_name = target_agent

            # capture tool calls using get_function_calls()
            for fc in event.get_function_calls():
                call = ToolCall(
                    tool_name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                    agent_name=current_agent_name,
                )
                tool_calls.append(call)
                logger.debug("Tool call: %s", fc.name)

            # capture function responses and update tool call results
            for fr in event.get_function_responses():
                for call in tool_calls:
                    if call.tool_name == fr.name and call.result is None:
                        call.result = fr.response
                        break

            # capture text responses from content parts
            if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        agent_text_parts.append(text)

            # capture token usage from usage_metadata
            if hasattr(event, "usage_metadata") and event.usage_metadata:
                usage = event.usage_metadata
                input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                output_tokens += getattr(usage, "candidates_token_count", 0) or 0
                thinking_tokens += getattr(usage, "thoughts_token_count", 0) or 0

            # detect if agent signaled conversation end via escalate action
            if (
                hasattr(event, "actions")
                and event.actions
                and getattr(event.actions, "escalate", False)
            ):
                terminal_state = "agent_ended"
                logger.debug("Agent escalated (ended conversation)")

        self._current_agent = current_agent_name

        # capture state snapshot from session
        state_snapshot = None
        if self._session and hasattr(self._session, "state"):
            state_snapshot = dict(self._session.state) if self._session.state else None

        return AgentResponse(
            content=" ".join(agent_text_parts),
            tool_calls=tool_calls,
            terminal_state=terminal_state,
            agent_name=current_agent_name,
            agent_transfers=list(self._agent_transfers),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            state_snapshot=state_snapshot,
        )

    def stop(self) -> None:
        """Clean up the ADK session."""
        self._runner = None
        self._session = None
        self._session_service = None
        self._mocks = None
