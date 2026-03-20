"""Runner: orchestrates the simulation loop."""

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .batch import BatchExecutor
from .mocks import MockToolkit
from .models import Scene
from .trace import StateSnapshot, ToolCall, Trace, Turn, TurnMetrics

if TYPE_CHECKING:
    pass

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
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        state_snapshot: dict[str, Any] | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.terminal_state = terminal_state
        self.agent_name = agent_name
        self.agent_transfers = agent_transfers or []
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens
        self.state_snapshot = state_snapshot


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

            # send to agent and measure latency
            start_time = time.perf_counter()
            response = app.send(user_message)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

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

            # record turn metrics (token counts from adapter, latency from here)
            trace.metrics.turns.append(
                TurnMetrics(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    thinking_tokens=response.thinking_tokens,
                    latency_ms=latency_ms,
                )
            )

            # record state snapshot if provided by adapter
            if response.state_snapshot:
                trace.state_snapshots.append(
                    StateSnapshot(
                        turn_number=turn_num + 1,
                        state=response.state_snapshot,
                        timestamp=datetime.now(UTC),
                    )
                )

            # collect agent transfers
            if response.agent_transfers:
                trace.agent_transfers.extend(response.agent_transfers)

            # agent hangs up - agent signaled conversation is over
            if response.terminal_state:
                trace.terminal_state = response.terminal_state
                logger.info("Scene %s: agent ended (%s)", scene.id, response.terminal_state)
                break

            # generate next user turn - simulator decides if conversation continues
            next_turn = simulator.next_turn(history)
            if next_turn is None:
                # simulator signaled conversation is done (goal achieved or natural end)
                trace.terminal_state = "completed"
                logger.info("Scene %s completed (simulator finished)", scene.id)
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


def simulate(
    app: AgentApp,
    scene: Scene,
    mocks: MockToolkit | None = None,
    simulator_backend: Any | None = None,
    simulator_model: str = "gpt-4o",
) -> Trace:
    """Run a simulation and return the trace (no evaluation).

    This is an alias for run() with clearer naming to indicate
    simulation-only behavior.

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
    return run(
        app=app,
        scene=scene,
        mocks=mocks,
        simulator_backend=simulator_backend,
        simulator_model=simulator_model,
    )


class _SimulationTask:
    """Internal task descriptor for batch simulation."""

    def __init__(self, scene: Scene, sim_index: int):
        self.scene = scene
        self.sim_index = sim_index


class _SimulationExecutor(BatchExecutor[_SimulationTask, Trace]):
    """Executor for running simulations in parallel."""

    def __init__(
        self,
        app: AgentApp,
        mocks: MockToolkit | None,
        simulator_model: str,
        storage: Any,
        tags: dict[str, str] | None,
        parallel: int = 1,
    ):
        super().__init__(parallel)
        self.app = app
        self.mocks = mocks
        self.simulator_model = simulator_model
        self.storage = storage
        self.tags = tags

    def execute_one(self, item: _SimulationTask) -> Trace:
        trace = simulate(
            app=self.app,
            scene=item.scene,
            mocks=self.mocks,
            simulator_model=self.simulator_model,
        )
        if self.storage:
            self.storage.save(
                trace=trace,
                scene=item.scene,
                sim_index=item.sim_index,
                tags=self.tags,
            )
        return trace


def simulate_batch(
    app: AgentApp,
    scenes: list[Scene] | str | Path,
    simulator_model: str = "gpt-4o",
    n_sims: int = 1,
    parallel: int = 1,
    mocks: MockToolkit | None = None,
    output: str | Path | None = None,
    tags: dict[str, str] | None = None,
) -> list[Trace]:
    """Run multiple simulations and return traces (no evaluation).

    Args:
        app: The agent application to test.
        scenes: List of Scene objects, or path to scene file/directory.
        simulator_model: Model name for the user simulator.
        n_sims: Number of simulations per scene.
        parallel: Number of parallel execution threads.
        mocks: Optional mock toolkit for tool responses.
        output: Optional path to save trace files.
        tags: Optional metadata tags.

    Returns:
        List of Trace objects from all simulations.
    """
    from .models import Scene as SceneModel

    if isinstance(scenes, (str, Path)):
        path = Path(scenes)
        if path.is_dir():
            scene_list = []
            for f in sorted(path.iterdir()):
                if f.suffix in (".yaml", ".yml", ".json"):
                    scene_list.append(SceneModel.from_file(f))
        else:
            scene_list = [SceneModel.from_file(path)]
    else:
        scene_list = scenes

    storage = None
    if output:
        from .storage import TraceStorage

        storage = TraceStorage(path=Path(output))

    tasks = [
        _SimulationTask(scene, sim_index) for scene in scene_list for sim_index in range(n_sims)
    ]

    executor = _SimulationExecutor(
        app=app,
        mocks=mocks,
        simulator_model=simulator_model,
        storage=storage,
        tags=tags,
        parallel=parallel,
    )

    return executor.run(tasks)
