"""Runner: orchestrates the simulation loop."""

import logging
from datetime import UTC, datetime
from typing import Any, Protocol

from .mocks import MockToolkit
from .models import Scene
from .trace import ToolCall, Trace, Turn

logger = logging.getLogger(__name__)


class AgentApp(Protocol):
    """Protocol for agent applications that understudy can drive.

    Implementations wrap the actual agent framework (ADK, LangGraph, etc.)
    and expose a simple send/receive interface.
    """

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the agent session."""
        ...

    def send(self, message: str) -> "AgentResponse":
        """Send a user message and get the agent's response."""
        ...

    def stop(self) -> None:
        """Tear down the agent session."""
        ...


class AgentResponse:
    """Response from the agent after processing a user message."""

    def __init__(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        terminal_state: str | None = None,
        agent_name: str | None = None,
        agent_transfers: list | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.terminal_state = terminal_state
        self.agent_name = agent_name
        self.agent_transfers = agent_transfers or []


class LiteLLMBackend:
    """LLM backend using litellm for unified provider access.

    Supports any model string that litellm supports:
    - OpenAI: "gpt-4o", "gpt-4o-mini"
    - Anthropic: "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"
    - Google: "gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro"
    - And many more providers.

    See https://docs.litellm.ai/docs/providers for full list.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def generate(self, prompt: str) -> str:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return content or ""


def run(
    app: AgentApp,
    scene: Scene,
    mocks: MockToolkit | None = None,
    simulator_backend: Any | None = None,
    simulator_model: str = "gpt-4o",
) -> Trace:
    """Run a scene against an agent app and return the trace.

    Args:
        app: The agent application to test.
        scene: The scene (conversation fixture) to run.
        mocks: Optional mock toolkit for tool responses.
        simulator_backend: LLM backend for the user simulator. If None, uses
            LiteLLMBackend with the specified model.
        simulator_model: Model name for the default LiteLLMBackend.

    Returns:
        A Trace recording everything that happened.
    """
    from .simulator import Simulator

    # mocks are optional - developer provides them if needed

    # set up simulator
    if simulator_backend is None:
        simulator_backend = LiteLLMBackend(model=simulator_model)

    simulator = Simulator(
        backend=simulator_backend,
        conversation_plan=scene.conversation_plan,
        persona_prompt=scene.persona.to_prompt(),
    )

    # initialize trace
    trace = Trace(
        scene_id=scene.id,
        started_at=datetime.now(UTC),
    )

    logger.info("Running scene: %s", scene.id)

    # start the agent
    app.start(mocks=mocks)

    try:
        # send starting prompt
        history: list[dict[str, str]] = []
        user_message = scene.starting_prompt

        for turn_num in range(scene.max_turns):
            logger.debug("Turn %d", turn_num + 1)
            # record user turn
            trace.turns.append(
                Turn(
                    role="user",
                    content=user_message,
                    timestamp=datetime.now(UTC),
                )
            )
            history.append({"role": "user", "content": user_message})

            # send to agent
            response = app.send(user_message)

            # record agent turn
            trace.turns.append(
                Turn(
                    role="agent",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    timestamp=datetime.now(UTC),
                    agent_name=response.agent_name,
                )
            )
            history.append({"role": "assistant", "content": response.content})

            # collect agent transfers
            if response.agent_transfers:
                trace.agent_transfers.extend(response.agent_transfers)

            # check for terminal state
            if response.terminal_state:
                trace.terminal_state = response.terminal_state
                logger.info("Scene %s completed: %s", scene.id, response.terminal_state)
                break

            # generate next user turn
            next_turn = simulator.next_turn(history)
            if next_turn is None:
                # simulator signaled conversation is done
                break

            user_message = next_turn
        else:
            # max turns reached without resolution
            trace.terminal_state = "max_turns_reached"
            logger.warning("Scene %s: max turns reached", scene.id)

    finally:
        app.stop()
        trace.finished_at = datetime.now(UTC)

    return trace
