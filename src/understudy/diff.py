"""Trace diffing for comparing agent behavior across versions."""

from dataclasses import dataclass, field
from typing import Any

from .trace import ToolCall, Trace


@dataclass
class ToolCallDiff:
    """Difference in a single tool call."""

    tool_name: str
    status: str  # "added", "removed", "changed", "unchanged"
    trace1_call: ToolCall | None = None
    trace2_call: ToolCall | None = None
    arg_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class TraceDiff:
    """Result of comparing two traces."""

    trace1_id: str
    trace2_id: str
    added_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)
    changed_calls: list[ToolCallDiff] = field(default_factory=list)
    unchanged_calls: list[str] = field(default_factory=list)
    terminal_state_changed: bool = False
    trace1_terminal: str | None = None
    trace2_terminal: str | None = None
    turn_count_diff: int = 0

    @property
    def has_changes(self) -> bool:
        """Return True if there are any differences."""
        return bool(
            self.added_tools
            or self.removed_tools
            or self.changed_calls
            or self.terminal_state_changed
        )

    @property
    def regression_warnings(self) -> list[str]:
        """Identify potential regressions.

        A regression is when:
        - A tool that was called is no longer called
        - Terminal state went from success to failure
        """
        warnings = []

        for tool in self.removed_tools:
            warnings.append(f"Tool '{tool}' was called before but not after")

        if (
            self.terminal_state_changed
            and self.trace1_terminal in ("completed", "done", "success")
            and self.trace2_terminal not in ("completed", "done", "success")
        ):
            warnings.append(
                f"Terminal state regressed: {self.trace1_terminal} -> {self.trace2_terminal}"
            )

        return warnings

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Diff: {self.trace1_id} vs {self.trace2_id}"]
        lines.append("=" * 50)

        if not self.has_changes:
            lines.append("No differences found.")
            return "\n".join(lines)

        if self.added_tools:
            lines.append(f"\nAdded tools ({len(self.added_tools)}):")
            for tool in self.added_tools:
                lines.append(f"  + {tool}")

        if self.removed_tools:
            lines.append(f"\nRemoved tools ({len(self.removed_tools)}):")
            for tool in self.removed_tools:
                lines.append(f"  - {tool}")

        if self.changed_calls:
            lines.append(f"\nChanged calls ({len(self.changed_calls)}):")
            for diff in self.changed_calls:
                lines.append(f"  ~ {diff.tool_name}")
                for key, (old, new) in diff.arg_changes.items():
                    lines.append(f"      {key}: {old} -> {new}")

        if self.terminal_state_changed:
            lines.append(f"\nTerminal state: {self.trace1_terminal} -> {self.trace2_terminal}")

        if self.turn_count_diff != 0:
            direction = "more" if self.turn_count_diff > 0 else "fewer"
            lines.append(f"\nTurns: {abs(self.turn_count_diff)} {direction} turns")

        warnings = self.regression_warnings
        if warnings:
            lines.append("\nPotential regressions:")
            for w in warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)


def diff_traces(trace1: Trace, trace2: Trace) -> TraceDiff:
    """Compare two traces and identify differences.

    Args:
        trace1: The baseline trace (before/expected)
        trace2: The candidate trace (after/actual)

    Returns:
        TraceDiff with all identified differences.
    """
    result = TraceDiff(
        trace1_id=trace1.scene_id,
        trace2_id=trace2.scene_id,
    )

    calls1 = trace1.call_sequence()
    calls2 = trace2.call_sequence()

    set1 = set(calls1)
    set2 = set(calls2)

    result.added_tools = sorted(set2 - set1)
    result.removed_tools = sorted(set1 - set2)
    result.unchanged_calls = sorted(set1 & set2)

    calls1_by_name: dict[str, list[ToolCall]] = {}
    for call in trace1.tool_calls:
        calls1_by_name.setdefault(call.tool_name, []).append(call)

    calls2_by_name: dict[str, list[ToolCall]] = {}
    for call in trace2.tool_calls:
        calls2_by_name.setdefault(call.tool_name, []).append(call)

    for tool_name in set1 & set2:
        list1 = calls1_by_name.get(tool_name, [])
        list2 = calls2_by_name.get(tool_name, [])

        for call1, call2 in zip(list1, list2, strict=False):
            arg_changes = _diff_arguments(call1.arguments, call2.arguments)
            if arg_changes:
                result.changed_calls.append(
                    ToolCallDiff(
                        tool_name=tool_name,
                        status="changed",
                        trace1_call=call1,
                        trace2_call=call2,
                        arg_changes=arg_changes,
                    )
                )

    if trace1.terminal_state != trace2.terminal_state:
        result.terminal_state_changed = True
        result.trace1_terminal = trace1.terminal_state
        result.trace2_terminal = trace2.terminal_state

    result.turn_count_diff = trace2.turn_count - trace1.turn_count

    return result


def _diff_arguments(args1: dict[str, Any], args2: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Find differences between two argument dicts."""
    changes = {}

    all_keys = set(args1.keys()) | set(args2.keys())
    for key in all_keys:
        val1 = args1.get(key)
        val2 = args2.get(key)
        if val1 != val2:
            changes[key] = (val1, val2)

    return changes


def diff_tool_sequences(seq1: list[str], seq2: list[str]) -> dict[str, Any]:
    """Compare two tool call sequences.

    Returns:
        Dict with sequence comparison information.
    """
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, seq1, seq2)
    ratio = matcher.ratio()

    operations = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            operations.append(
                {
                    "op": "equal",
                    "tools": seq1[i1:i2],
                }
            )
        elif tag == "replace":
            operations.append(
                {
                    "op": "replace",
                    "before": seq1[i1:i2],
                    "after": seq2[j1:j2],
                }
            )
        elif tag == "delete":
            operations.append(
                {
                    "op": "delete",
                    "tools": seq1[i1:i2],
                }
            )
        elif tag == "insert":
            operations.append(
                {
                    "op": "insert",
                    "tools": seq2[j1:j2],
                }
            )

    return {
        "similarity": ratio,
        "operations": operations,
        "before_length": len(seq1),
        "after_length": len(seq2),
    }
