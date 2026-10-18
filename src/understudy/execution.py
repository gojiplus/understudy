"""Isolated single-turn execution, state fixtures, and checkpoint capture."""

from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .mocks import MockToolkit
from .runner import AgentApp, AgentResponse
from .trace import Trace, Turn, TurnMetrics


class UnsupportedCapabilityError(ValueError):
    """The adapter cannot faithfully perform the requested operation."""


class AdapterCapabilities(BaseModel):
    """Operations and observations an adapter can guarantee."""

    tools: bool = False
    agents: bool = False
    nested: bool = False
    state: bool = False
    checkpoints: bool = False
    interception: bool = False


class StateFixture(BaseModel):
    """Synthetic conversation and state, without a claim of runtime equivalence."""

    state: dict[str, Any] = Field(default_factory=dict)
    history: list[Turn] = Field(default_factory=list)
    tool_state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("history")
    @classmethod
    def completed_history(cls, history: list[Turn]) -> list[Turn]:
        """Require a completed input/response boundary for synthetic history."""
        if len(history) % 2 or any(
            t.role != ("user" if i % 2 == 0 else "agent") for i, t in enumerate(history)
        ):
            raise ValueError(
                "History must contain complete alternating user/agent exchanges"
            )
        return history


class Checkpoint(BaseModel):
    """Versioned adapter state captured at a completed turn boundary."""

    schema_version: Literal[1] = 1
    adapter: str
    framework_version: str
    app_id: str
    after_turn: int = Field(default=0, ge=0)
    state: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    history: list[Turn] = Field(default_factory=list)
    tool_state: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Write a portable checkpoint as JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        """Validate and read a checkpoint from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class TurnResult(BaseModel):
    """New-turn observations, separate from the history used to initialize it."""

    trace: Trace
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    checkpoint: Checkpoint | None = None

    @property
    def state_changes(self) -> dict[str, dict[str, Any]]:
        """Return changed top-level keys, distinguishing missing values from null."""
        before, after = self.state_before or {}, self.state_after or {}
        return {
            key: {
                "before_present": key in before,
                "before": before.get(key),
                "after_present": key in after,
                "after": after.get(key),
            }
            for key in before.keys() | after.keys()
            if key not in before or key not in after or before[key] != after[key]
        }


def capabilities(app: AgentApp) -> AdapterCapabilities:
    """Read declared capabilities; custom adapters opt in explicitly."""
    return getattr(app, "capabilities", AdapterCapabilities())


def response_trace(response: AgentResponse, message: str, app: AgentApp) -> Trace:
    """Build a trace containing exactly one input and its resulting execution."""
    caps = capabilities(app)
    trace = Trace(
        scene_id="single-turn",
        turns=[
            Turn(role="user", content=message),
            Turn(
                role="agent",
                content=response.content,
                tool_calls=response.tool_calls,
                agent_name=response.agent_name,
            ),
        ],
        terminal_state=response.terminal_state,
        agent_transfers=response.agent_transfers,
        executions=response.executions,
        tools_observed=caps.tools,
        agents_observed=caps.agents,
    )
    trace.metrics.turns.append(
        TurnMetrics(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            thinking_tokens=response.thinking_tokens,
        )
    )
    return trace


def _prepare(
    app: Any,
    mocks: MockToolkit | None,
    checkpoint: Checkpoint | None,
    fixture: StateFixture | None,
    state_patch: dict[str, Any] | None,
    capture: bool,
) -> None:
    caps = capabilities(app)
    if checkpoint is not None and fixture is not None:
        raise ValueError("Choose a checkpoint or a state fixture, not both")
    if (checkpoint is not None or capture) and not caps.checkpoints:
        raise UnsupportedCapabilityError("Adapter does not support checkpoints")
    if (fixture is not None or state_patch is not None) and not caps.state:
        raise UnsupportedCapabilityError("Adapter does not support state injection")
    if mocks is not None and not caps.interception:
        raise UnsupportedCapabilityError("Adapter cannot intercept tools")
    if checkpoint is not None:
        app.validate_checkpoint(checkpoint)
    if mocks is not None:
        initial = checkpoint if checkpoint is not None else fixture
        if initial is not None:
            mocks.state.clear()
            mocks.state.update(deepcopy(initial.tool_state))
    app.start(mocks=mocks)
    if checkpoint is not None:
        app.restore(checkpoint.model_copy(deep=True))
    elif fixture is not None:
        app.inject(fixture.model_copy(deep=True))
    if state_patch is not None:
        app.patch_state(deepcopy(state_patch))


def run_turn(
    app_factory: Callable[[], Any],
    message: str | None = None,
    *,
    checkpoint: Checkpoint | None = None,
    fixture: StateFixture | None = None,
    state_patch: dict[str, Any] | None = None,
    mocks_factory: Callable[[], MockToolkit] | None = None,
    next_message: Callable[[list[Turn]], str] | None = None,
    capture_checkpoint: bool = False,
) -> TurnResult:
    """Initialize an isolated app, execute one input, and collect only new activity.

    Args:
        app_factory: Creates a fresh adapter and runtime for this test.
        message: Explicit next user input, mutually exclusive with next_message.
        checkpoint: Captured state to fork without executing prior calls.
        fixture: Synthetic state/history to inject instead of a checkpoint.
        state_patch: Top-level state updates applied after initialization.
        mocks_factory: Creates fresh handlers and mock-service state.
        next_message: Generates one user input from a copy of the prior history.
        capture_checkpoint: Capture a portable checkpoint after this turn.

    Returns:
        New-turn trace, observed state, and optionally a checkpoint.

    Raises:
        ValueError: If neither or both input sources are supplied.
        TypeError: If the input generator does not return a string.
    """
    if (message is None) == (next_message is None):
        raise ValueError("Supply exactly one of message or next_message")
    app = app_factory()
    mocks = mocks_factory() if mocks_factory is not None else None
    initial = checkpoint if checkpoint is not None else fixture
    history = deepcopy(initial.history) if initial is not None else []
    try:
        _prepare(app, mocks, checkpoint, fixture, state_patch, capture_checkpoint)
        state_before = deepcopy(app.get_state()) if capabilities(app).state else None
        if next_message is not None:
            message = next_message(deepcopy(history))
        if not isinstance(message, str):
            raise TypeError("The next user message must be a string")
        started = datetime.now(UTC)
        response = app.send(message)
        trace = response_trace(response, message, app)
        trace.started_at, trace.finished_at = started, datetime.now(UTC)
        trace.metadata["starting_point"] = {
            "mode": "checkpoint" if checkpoint else "fixture" if fixture else "fresh",
            "after_turn": checkpoint.after_turn if checkpoint else len(history) // 2,
        }
        trace.metrics.turns[0].latency_ms = int(
            (trace.finished_at - started).total_seconds() * 1000
        )
        after = deepcopy(app.get_state()) if capabilities(app).state else None
        saved = None
        if capture_checkpoint:
            saved = app.checkpoint()
            saved.history = history + deepcopy(trace.turns)
            saved.after_turn = (
                checkpoint.after_turn if checkpoint else len(history) // 2
            ) + 1
            saved.tool_state = deepcopy(mocks.state) if mocks is not None else {}
        return TurnResult(
            trace=trace, state_before=state_before, state_after=after, checkpoint=saved
        )
    finally:
        app.stop()


def replay_to(
    app_factory: Callable[[], Any],
    trace: Trace,
    after_turn: int,
    *,
    mocks_factory: Callable[[], MockToolkit] | None = None,
    require_match: bool = True,
) -> Checkpoint:
    """Re-execute a recorded prefix and capture its resulting state.

    Args:
        app_factory: Creates an isolated adapter supporting checkpoints.
        trace: Recorded inputs and reference outputs.
        after_turn: Number of complete input/response pairs to execute; zero is valid.
        mocks_factory: Supplies tools before execution, not after it.
        require_match: Stop if text or tool names/arguments/results diverge.

    Returns:
        A checkpoint of the newly executed prefix, not the original runtime.

    Raises:
        ValueError: If the boundary, history, or replayed behavior is invalid.
    """
    pairs = list(zip(trace.turns[::2], trace.turns[1::2], strict=True))
    if not 0 <= after_turn <= len(pairs):
        raise ValueError("after_turn is outside the recorded conversation")
    if any(u.role != "user" or a.role != "agent" for u, a in pairs):
        raise ValueError("Replay requires complete alternating user/agent turns")
    app = app_factory()
    mocks = mocks_factory() if mocks_factory else None
    history = []
    try:
        _prepare(app, mocks, None, None, None, True)
        for index, (user, expected) in enumerate(pairs[:after_turn], 1):
            response = app.send(user.content)
            actual = response_trace(response, user.content, app).turns[-1]
            observed_calls = [
                (c.tool_name, c.arguments, c.result) for c in actual.tool_calls
            ]
            expected_calls = [
                (c.tool_name, c.arguments, c.result) for c in expected.tool_calls
            ]
            if require_match and (
                actual.content != expected.content or observed_calls != expected_calls
            ):
                raise ValueError(f"Replay diverged at turn {index}")
            history.extend([user.model_copy(deep=True), actual])
            if response.terminal_state and index < after_turn:
                raise ValueError(
                    f"Agent ended before the requested boundary at turn {index}"
                )
        saved = app.checkpoint()
        saved.after_turn, saved.history = after_turn, history
        saved.tool_state = deepcopy(mocks.state) if mocks else {}
        return saved
    finally:
        app.stop()
