"""ADK adapter: wraps Google ADK agents for use with understudy."""

from datetime import UTC, datetime
from typing import Any

from ..mocks import MockToolkit
from ..runner import AgentApp, AgentResponse
from ..trace import AgentTransfer, ToolCall


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
        self._mocks: MockToolkit | None = None
        self._current_agent: str | None = None
        self._agent_transfers: list[AgentTransfer] = []

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the ADK session."""
        try:
            from google.adk import Runner
            from google.adk.sessions import InMemorySessionService
        except ImportError as e:
            raise ImportError(
                "google-adk package required. Install with: pip install understudy[adk]"
            ) from e
        import uuid

        self._mocks = mocks
        self._current_agent = getattr(self.agent, "name", None)
        self._agent_transfers = []
        self._session_id = self.session_id or str(uuid.uuid4())

        session_service = InMemorySessionService()
        if mocks:
            self.agent.before_tool_callback = _create_mock_callback(mocks)

        self._runner = Runner(
            agent=self.agent,
            app_name="understudy_test",
            session_service=session_service,
        )
        self._session = session_service.create_session_sync(
            app_name="understudy_test",
            user_id="understudy_user",
            session_id=self._session_id,
        )

    def send(self, message: str) -> AgentResponse:
        """Send a user message to the ADK agent and capture the response."""
        try:
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-adk package required. Install with: pip install understudy[adk]"
            ) from e

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )

        tool_calls: list[ToolCall] = []
        agent_text_parts: list[str] = []
        terminal_state: str | None = None
        current_agent_name = self._current_agent

        for event in self._runner.run(
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

                        # check for terminal state markers
                        # convention: agent emits "TERMINAL_STATE: <state>"
                        if "TERMINAL_STATE:" in text:
                            state = text.split("TERMINAL_STATE:")[-1].strip()
                            terminal_state = state.split()[0].strip()

        self._current_agent = current_agent_name

        response = AgentResponse(
            content=" ".join(agent_text_parts),
            tool_calls=tool_calls,
            terminal_state=terminal_state,
        )
        response.agent_name = current_agent_name
        response.agent_transfers = list(self._agent_transfers)
        return response

    def stop(self) -> None:
        """Clean up the ADK session."""
        self._runner = None
        self._session = None
        self._mocks = None
