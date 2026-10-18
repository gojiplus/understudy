"""ADK adapter: wraps Google ADK agents for use with understudy."""

import asyncio
import logging
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

from ..execution import (
    AdapterCapabilities,
    Checkpoint,
    StateFixture,
    UnsupportedCapabilityError,
)
from ..instrumentation import ExecutionRecorder
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


class ADKApp(AgentApp):
    """Wraps a Google ADK Agent for use with understudy.

    Usage:
        from google.adk import Agent
        from understudy.adk import ADKApp

        agent = Agent(model="gemini-2.5-flash", name="my_agent", ...)
        app = ADKApp(agent=agent)
        trace = run(app, scene)
    """

    capabilities = AdapterCapabilities(
        tools=True,
        agents=True,
        nested=True,
        state=True,
        checkpoints=True,
        interception=True,
    )

    def __init__(
        self, agent: Any, session_id: str | None = None, *, app_id: str | None = None
    ):
        """Wrap a google-adk agent.

        Args:
            agent: A google.adk.Agent instance.
            session_id: Optional session ID. If None, a random one is generated.
            app_id: Application revision identifier checked when restoring fixtures.
        """
        self.agent = agent
        self.app_id = app_id or getattr(agent, "name", "adk")
        self._loop: Any = None
        self._recorder = ExecutionRecorder()
        self._pending_tools = False
        self._plugin: Any = None
        self.session_id = session_id
        self._runner = None
        self._session: Any = None
        self._session_service: Any = None
        self._mocks: MockToolkit | None = None
        self._current_agent: str | None = None
        self._agent_transfers: list[AgentTransfer] = []

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the ADK session."""
        self._loop = asyncio.Runner()
        try:
            self._loop.run(self._start_async(mocks))
        except BaseException:
            self.stop()
            raise

    async def _start_async(self, mocks: MockToolkit | None = None) -> None:
        """Async implementation of start."""
        try:
            from google.adk import Runner
            from google.adk.apps import App
            from google.adk.sessions import InMemorySessionService

            from .instrumentation import TestPlugin
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

        logger.debug(
            "Starting ADK session %s for agent %s",
            self._session_id,
            self._current_agent,
        )

        self._session_service = InMemorySessionService()
        self._recorder = ExecutionRecorder()
        self._pending_tools = False
        self._plugin = TestPlugin(self._recorder, mocks)
        self._runner = Runner(
            app=App(
                name="understudy_test", root_agent=self.agent, plugins=[self._plugin]
            ),
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
        if self._loop is None:
            raise RuntimeError("ADK session is not open")
        return self._loop.run(self._send_async(message))

    async def _send_async(self, message: str) -> AgentResponse:
        """Async implementation of send."""
        try:
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "google-adk package required. Install with: pip install understudy[adk]"
            ) from e

        # _runner and _session are None until the session is opened. Without
        # this, calling out of order surfaced as
        # "AttributeError: 'NoneType' object has no attribute 'run_async'",
        # which names neither the cause nor the fix.
        if self._runner is None or self._session is None:
            raise RuntimeError(
                "ADK session is not open; use this adapter as a context manager "
                "or call start() before sending a message."
            )

        logger.debug("Sending message: %s", message[:100])

        user_content = types.Content(
            role="user",
            parts=[types.Part(text=message)],
        )

        self._pending_tools = False
        self._recorder.executions.clear()
        self._recorder.by_id.clear()
        tool_calls: list[ToolCall] = []
        self._agent_transfers = []
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
            if event.long_running_tool_ids:
                self._pending_tools = True
            # track agent attribution from event.author
            if hasattr(event, "author") and event.author:
                event_agent = event.author
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

            scope = (
                self._recorder.observe(event.node_info.path, event.invocation_id)
                if event.node_info.path
                else None
            )
            # capture tool calls using get_function_calls()
            for fc in event.get_function_calls():
                if not fc.name:
                    # Unnameable in a report, and ADK only omits the name on
                    # malformed calls. Dropping beats an anonymous row.
                    logger.debug("Skipping function call with no name")
                    continue
                call = ToolCall(
                    tool_name=fc.name,
                    arguments=dict(fc.args) if fc.args else {},
                    agent_name=event.author,
                    call_id=fc.id,
                    execution_id=scope.id if scope else None,
                    agent_path=scope.agent_path if scope else (),
                )
                tool_calls.append(call)
                logger.debug("Tool call: %s", fc.name)

            # capture function responses and update tool call results
            for fr in event.get_function_responses():
                for call in tool_calls:
                    if (
                        call.execution_id == (scope.id if scope else None)
                        and call.agent_name == event.author
                        and (
                            call.call_id == fr.id
                            if fr.id
                            else call.tool_name == fr.name
                        )
                        and call.result is None
                    ):
                        call.result = fr.response
                        break

            if event.content:
                agent_text_parts.extend(
                    part.text
                    for part in event.content.parts or []
                    if part.text and not part.thought
                )
            if event.usage_metadata:
                usage = event.usage_metadata
                input_tokens += usage.prompt_token_count or 0
                output_tokens += usage.candidates_token_count or 0
                thinking_tokens += usage.thoughts_token_count or 0
            if event.actions.escalate:
                terminal_state = "agent_ended"

        self._current_agent = current_agent_name
        self._pending_tools = self._pending_tools or any(
            call.result is None for call in tool_calls
        )
        if self._pending_tools:
            terminal_state = "interrupted"
        self._recorder.finish()
        state_snapshot = await self._state_async()

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
            executions=deepcopy(self._recorder.executions),
        )

    def _require_session(self) -> None:
        if self._loop is None or self._session is None:
            raise RuntimeError("ADK session is not open")

    async def _state_async(self) -> dict[str, Any]:
        self._session = await self._session_service.get_session(
            app_name="understudy_test",
            user_id="understudy_user",
            session_id=self._session.id,
        )
        return deepcopy(self._session.state)

    def get_state(self) -> dict[str, Any]:
        """Read the session's current managed state."""
        self._require_session()
        return self._loop.run(self._state_async())

    def validate_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Reject checkpoints from a different framework or application revision."""
        if (checkpoint.adapter, checkpoint.framework_version, checkpoint.app_id) != (
            "adk",
            version("google-adk"),
            self.app_id,
        ):
            raise ValueError(
                "Checkpoint adapter, framework version, or app_id does not match"
            )

    def checkpoint(self) -> Checkpoint:
        """Capture session state and native event history at a completed boundary."""
        state = self.get_state()
        if self._pending_tools:
            raise UnsupportedCapabilityError(
                "Checkpoint capture requires a completed turn "
                "without pending or long-running tools"
            )
        if any(event.actions.artifact_delta for event in self._session.events):
            raise UnsupportedCapabilityError(
                "Artifact storage is not included in ADK checkpoints"
            )
        return Checkpoint(
            adapter="adk",
            framework_version=version("google-adk"),
            app_id=self.app_id,
            state=state,
            payload={
                "events": [e.model_dump(mode="json") for e in self._session.events]
            },
        )

    async def _append_events(
        self, records: list[dict[str, Any]], state: dict[str, Any]
    ) -> None:
        from google.adk.events import Event, EventActions

        for record in records:
            event = Event.model_validate(record)
            await self._session_service.append_event(session=self._session, event=event)
        await self._session_service.append_event(
            session=self._session,
            event=Event(author="user", actions=EventActions(state_delta=state)),
        )
        await self._state_async()

    def restore(self, checkpoint: Checkpoint) -> None:
        """Restore native events and state into this fresh isolated session."""
        self.validate_checkpoint(checkpoint)
        self._require_session()
        self._loop.run(
            self._append_events(
                deepcopy(checkpoint.payload["events"]), deepcopy(checkpoint.state)
            )
        )

    def inject(self, fixture: StateFixture) -> None:
        """Seed synthetic history and managed state without executing the agent."""
        from google.adk.events import Event
        from google.genai import types

        self._require_session()
        events = []
        for turn in fixture.history:
            parts = [types.Part(text=turn.content)] if turn.content else []
            parts.extend(
                types.Part(
                    function_call=types.FunctionCall(
                        name=call.tool_name,
                        args=call.arguments,
                        id=call.call_id,
                    )
                )
                for call in turn.tool_calls
            )
            author = (
                "user" if turn.role == "user" else (turn.agent_name or self.agent.name)
            )
            role = "user" if turn.role == "user" else "model"
            events.append(
                Event(
                    author=author, content=types.Content(role=role, parts=parts)
                ).model_dump(mode="json")
            )
            if turn.tool_calls:
                results = [
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=c.tool_name,
                            id=c.call_id,
                            response=c.result
                            if isinstance(c.result, dict)
                            else {"result": c.result},
                        )
                    )
                    for c in turn.tool_calls
                ]
                events.append(
                    Event(
                        author=author, content=types.Content(role="user", parts=results)
                    ).model_dump(mode="json")
                )
        self._loop.run(self._append_events(events, deepcopy(fixture.state)))

    def patch_state(self, patch: dict[str, Any]) -> None:
        """Apply top-level updates using ADK's managed event mechanism."""
        self._require_session()
        self._loop.run(self._append_events([], deepcopy(patch)))

    def stop(self) -> None:
        """Close framework resources and the session's event loop."""
        if self._loop is not None:
            try:
                if self._runner is not None:
                    self._loop.run(self._runner.close())
            finally:
                self._loop.close()
                self._loop = None
        self._runner = None
        self._session = None
        self._session_service = None
        self._mocks = None
