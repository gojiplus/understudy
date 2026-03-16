"""Trace replay for testing agent behavior with recorded inputs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Persona, Scene
from .runner import AgentApp
from .trace import ToolCall, Trace, Turn


@dataclass
class ReplayResult:
    """Result of replaying a trace against a new agent."""

    original_trace: Trace
    new_trace: Trace
    matched_responses: int
    total_turns: int
    diverged_at_turn: int | None = None
    errors: list[str] | None = None

    @property
    def fully_matched(self) -> bool:
        """True if the replay produced identical behavior."""
        return self.matched_responses == self.total_turns

    @property
    def match_rate(self) -> float:
        """Fraction of turns that matched."""
        return self.matched_responses / self.total_turns if self.total_turns > 0 else 0.0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Replay: {self.original_trace.scene_id}"]
        lines.append("=" * 40)
        rate = f"{self.match_rate:.1%} ({self.matched_responses}/{self.total_turns})"
        lines.append(f"Match rate: {rate}")

        if self.diverged_at_turn is not None:
            lines.append(f"Diverged at turn: {self.diverged_at_turn}")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  - {err}")

        return "\n".join(lines)


def replay(
    trace: Trace,
    app: AgentApp,
    mocks: Any | None = None,
    compare_tool_calls: bool = True,
) -> ReplayResult:
    """Replay a trace against a new agent version.

    This function:
    1. Extracts user messages from the original trace
    2. Sends them to the new agent in order
    3. Compares the agent's responses and tool calls

    Args:
        trace: The original trace to replay
        app: The new agent app to test
        mocks: Optional mock toolkit for tool calls
        compare_tool_calls: Whether to compare tool calls (not just messages)

    Returns:
        ReplayResult with comparison data.
    """
    user_messages = [turn.content for turn in trace.turns if turn.role == "user"]

    new_turns: list[Turn] = []
    matched = 0
    diverged_at = None
    errors = []

    try:
        app.start(mocks=mocks)

        for i, message in enumerate(user_messages):
            try:
                response = app.send(message)

                new_tool_calls = []
                for tc in response.tool_calls or []:
                    if isinstance(tc, ToolCall):
                        new_tool_calls.append(tc)
                    elif isinstance(tc, dict):
                        new_tool_calls.append(
                            ToolCall(tool_name=tc["name"], arguments=tc.get("arguments", {}))
                        )

                new_turn = Turn(
                    role="agent",
                    content=response.content,
                    tool_calls=new_tool_calls,
                )
                new_turns.append(Turn(role="user", content=message))
                new_turns.append(new_turn)

                original_agent_turn = _get_agent_turn_after_user(trace, i)
                if original_agent_turn and _turns_match(
                    original_agent_turn, new_turn, compare_tool_calls
                ):
                    matched += 1
                elif diverged_at is None:
                    diverged_at = i

                if response.terminal_state:
                    break

            except Exception as e:
                errors.append(f"Turn {i}: {e}")
                if diverged_at is None:
                    diverged_at = i
                break

    finally:
        import contextlib

        with contextlib.suppress(Exception):
            app.stop()

    new_trace = Trace(
        scene_id=trace.scene_id,
        turns=new_turns,
        terminal_state=trace.terminal_state,
    )

    return ReplayResult(
        original_trace=trace,
        new_trace=new_trace,
        matched_responses=matched,
        total_turns=len(user_messages),
        diverged_at_turn=diverged_at,
        errors=errors if errors else None,
    )


def _get_agent_turn_after_user(trace: Trace, user_turn_index: int) -> Turn | None:
    """Get the agent turn that follows the nth user turn."""
    user_count = 0
    for i, turn in enumerate(trace.turns):
        if turn.role == "user":
            if user_count == user_turn_index:
                if i + 1 < len(trace.turns) and trace.turns[i + 1].role == "agent":
                    return trace.turns[i + 1]
                return None
            user_count += 1
    return None


def _turns_match(original: Turn, new: Turn, compare_tool_calls: bool) -> bool:
    """Check if two turns match."""
    if compare_tool_calls:
        orig_tools = [tc.tool_name for tc in original.tool_calls]
        new_tools = [tc.tool_name for tc in new.tool_calls]
        return orig_tools == new_tools

    return True


def load_trace(path: str | Path) -> Trace:
    """Load a trace from a JSON file."""
    import json

    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    if "trace" in data:
        data = data["trace"]

    return Trace.model_validate(data)


def create_replay_scene(trace: Trace) -> Scene:
    """Create a Scene from a trace for replay testing.

    This allows using the standard suite runner with recorded data.
    """
    user_messages = [turn.content for turn in trace.turns if turn.role == "user"]

    return Scene(
        id=f"replay_{trace.scene_id}",
        description=f"Replay of {trace.scene_id}",
        starting_prompt=user_messages[0] if user_messages else "",
        conversation_plan="Replay recorded conversation",
        persona=Persona(
            description="Replay persona - sends recorded messages",
            behaviors=["Sends pre-recorded user messages in sequence"],
        ),
    )
