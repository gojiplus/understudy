"""Trace: the source of truth for what happened during a rehearsal."""

from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation recorded from the agent."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    timestamp: datetime | None = None
    error: str | None = None
    agent_name: str | None = None


class AgentTransfer(BaseModel):
    """Records a handoff between agents in multi-agent systems."""

    from_agent: str
    to_agent: str
    timestamp: datetime | None = None


class Turn(BaseModel):
    """One turn in the conversation."""

    role: str  # "user" or "agent"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime | None = None
    agent_name: str | None = None


class Trace(BaseModel):
    """The full execution trace of a rehearsal.

    This is the source of truth. Assert against this, not the prose.
    """

    scene_id: str
    turns: list[Turn] = Field(default_factory=list)
    terminal_state: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    agent_transfers: list[AgentTransfer] = Field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls across all turns, in order."""
        calls = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def duration(self) -> timedelta | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    def called(self, tool_name: str, **kwargs: Any) -> bool:
        """Check if a tool was called, optionally with specific arguments.

        Examples:
            trace.called("lookup_order")
            trace.called("lookup_order", order_id="ORD-10027")
        """
        for call in self.tool_calls:
            if call.tool_name == tool_name:
                if not kwargs:
                    return True
                if all(call.arguments.get(k) == v for k, v in kwargs.items()):
                    return True
        return False

    def calls_to(self, tool_name: str) -> list[ToolCall]:
        """Get all calls to a specific tool."""
        return [c for c in self.tool_calls if c.tool_name == tool_name]

    def call_sequence(self) -> list[str]:
        """Ordered list of tool names called."""
        return [c.tool_name for c in self.tool_calls]

    @property
    def events(self) -> list[dict[str, Any]]:
        """State transitions, handoffs, escalations extracted from trace."""
        evts = []
        for turn in self.turns:
            for call in turn.tool_calls:
                if call.tool_name in (
                    "escalate_to_human",
                    "transfer_to_agent",
                    "handoff",
                ):
                    evts.append(
                        {
                            "type": "escalation",
                            "tool": call.tool_name,
                            "args": call.arguments,
                        }
                    )
        return evts

    def conversation_text(self) -> str:
        """Render the conversation as readable text (for judge input)."""
        lines = []
        for turn in self.turns:
            prefix = "[USER]" if turn.role == "user" else "[AGENT]"
            if turn.agent_name:
                prefix = f"[{turn.agent_name.upper()}]"
            lines.append(f"{prefix}: {turn.content}")
            for call in turn.tool_calls:
                lines.append(f"  -> {call.tool_name}({call.arguments})")
                if call.result is not None:
                    result_str = str(call.result)
                    if len(result_str) > 200:
                        result_str = result_str[:200] + "..."
                    lines.append(f"  <- {result_str}")
        return "\n".join(lines)

    def agents_invoked(self) -> list[str]:
        """Get list of unique agent names that participated in the conversation."""
        agents = set()
        for turn in self.turns:
            if turn.agent_name:
                agents.add(turn.agent_name)
        for call in self.tool_calls:
            if call.agent_name:
                agents.add(call.agent_name)
        return sorted(agents)

    def agent_called(self, agent: str, tool: str) -> bool:
        """Check if a specific agent called a specific tool."""
        return any(call.agent_name == agent and call.tool_name == tool for call in self.tool_calls)

    def calls_by_agent(self, agent: str) -> list[ToolCall]:
        """Get all tool calls made by a specific agent."""
        return [c for c in self.tool_calls if c.agent_name == agent]
