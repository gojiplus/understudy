"""Check: validate a trace against scene expectations."""

from dataclasses import dataclass, field

from .models import Expectations
from .trace import Trace


@dataclass
class CheckResult:
    """Result of checking a trace against expectations."""

    checks: list["CheckItem"] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list["CheckItem"]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.label}: {c.detail}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_pass = sum(1 for c in self.checks if c.passed)
        return f"CheckResult({n_pass}/{len(self.checks)} passed)"


@dataclass
class CheckItem:
    """A single check result."""

    label: str
    passed: bool
    detail: str


def check(trace: Trace, expectations: Expectations) -> CheckResult:
    """Validate a trace against expectations.

    Args:
        trace: The execution trace from a rehearsal.
        expectations: The expectations from a scene.

    Returns:
        A CheckResult with individual check outcomes.
    """
    result = CheckResult()
    called_tools = set(trace.call_sequence())

    # required tools
    for tool in expectations.required_tools:
        result.checks.append(
            CheckItem(
                label="required_tool",
                passed=tool in called_tools,
                detail=f"{tool} {'called' if tool in called_tools else 'NOT called'}",
            )
        )

    # forbidden tools
    for tool in expectations.forbidden_tools:
        was_called = tool in called_tools
        result.checks.append(
            CheckItem(
                label="forbidden_tool",
                passed=not was_called,
                detail=f"{tool} {'CALLED (violation)' if was_called else 'not called'}",
            )
        )

    # terminal state
    if expectations.allowed_terminal_states:
        in_allowed = trace.terminal_state in expectations.allowed_terminal_states
        result.checks.append(
            CheckItem(
                label="terminal_state",
                passed=in_allowed,
                detail=(
                    f"{trace.terminal_state} ({'allowed' if in_allowed else 'NOT in allowed'})"
                ),
            )
        )

    if expectations.forbidden_terminal_states:
        in_forbidden = trace.terminal_state in expectations.forbidden_terminal_states
        result.checks.append(
            CheckItem(
                label="forbidden_terminal_state",
                passed=not in_forbidden,
                detail=(
                    f"{trace.terminal_state} "
                    f"{'FORBIDDEN (violation)' if in_forbidden else 'not forbidden'}"
                ),
            )
        )

    # required agents
    invoked_agents = set(trace.agents_invoked())
    for agent in expectations.required_agents:
        result.checks.append(
            CheckItem(
                label="required_agent",
                passed=agent in invoked_agents,
                detail=f"{agent} {'invoked' if agent in invoked_agents else 'NOT invoked'}",
            )
        )

    # forbidden agents
    for agent in expectations.forbidden_agents:
        was_invoked = agent in invoked_agents
        result.checks.append(
            CheckItem(
                label="forbidden_agent",
                passed=not was_invoked,
                detail=f"{agent} {'INVOKED (violation)' if was_invoked else 'not invoked'}",
            )
        )

    # required agent tools
    for agent, tools in expectations.required_agent_tools.items():
        for tool in tools:
            called = trace.agent_called(agent, tool)
            result.checks.append(
                CheckItem(
                    label="required_agent_tool",
                    passed=called,
                    detail=f"{agent}.{tool} {'called' if called else 'NOT called'}",
                )
            )

    return result
