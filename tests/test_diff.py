"""Tests for trace diffing functionality."""

import pytest

from understudy import Trace, Turn, ToolCall, diff_traces, TraceDiff, diff_tool_sequences


class TestDiffTraces:
    def _make_trace(
        self,
        scene_id: str = "test",
        tool_calls: list[tuple[str, dict]] | None = None,
        terminal_state: str | None = "completed",
    ) -> Trace:
        calls = [ToolCall(tool_name=name, arguments=args) for name, args in (tool_calls or [])]
        return Trace(
            scene_id=scene_id,
            turns=[
                Turn(role="user", content="Hello"),
                Turn(role="agent", content="Hi", tool_calls=calls),
            ],
            terminal_state=terminal_state,
        )

    def test_identical_traces_no_diff(self):
        trace1 = self._make_trace(tool_calls=[("lookup_order", {"id": "123"})])
        trace2 = self._make_trace(tool_calls=[("lookup_order", {"id": "123"})])

        diff = diff_traces(trace1, trace2)

        assert not diff.has_changes
        assert diff.added_tools == []
        assert diff.removed_tools == []
        assert diff.changed_calls == []

    def test_added_tool(self):
        trace1 = self._make_trace(tool_calls=[("lookup_order", {})])
        trace2 = self._make_trace(tool_calls=[("lookup_order", {}), ("create_return", {})])

        diff = diff_traces(trace1, trace2)

        assert diff.has_changes
        assert "create_return" in diff.added_tools
        assert diff.removed_tools == []

    def test_removed_tool(self):
        trace1 = self._make_trace(tool_calls=[("lookup_order", {}), ("create_return", {})])
        trace2 = self._make_trace(tool_calls=[("lookup_order", {})])

        diff = diff_traces(trace1, trace2)

        assert diff.has_changes
        assert "create_return" in diff.removed_tools
        assert diff.added_tools == []

    def test_changed_arguments(self):
        trace1 = self._make_trace(tool_calls=[("lookup_order", {"id": "123"})])
        trace2 = self._make_trace(tool_calls=[("lookup_order", {"id": "456"})])

        diff = diff_traces(trace1, trace2)

        assert diff.has_changes
        assert len(diff.changed_calls) == 1
        assert diff.changed_calls[0].tool_name == "lookup_order"
        assert "id" in diff.changed_calls[0].arg_changes

    def test_terminal_state_changed(self):
        trace1 = self._make_trace(terminal_state="completed")
        trace2 = self._make_trace(terminal_state="failed")

        diff = diff_traces(trace1, trace2)

        assert diff.has_changes
        assert diff.terminal_state_changed
        assert diff.trace1_terminal == "completed"
        assert diff.trace2_terminal == "failed"

    def test_regression_warnings_removed_tool(self):
        trace1 = self._make_trace(tool_calls=[("lookup_order", {}), ("create_return", {})])
        trace2 = self._make_trace(tool_calls=[("lookup_order", {})])

        diff = diff_traces(trace1, trace2)

        warnings = diff.regression_warnings
        assert len(warnings) == 1
        assert "create_return" in warnings[0]

    def test_regression_warnings_terminal_state(self):
        trace1 = self._make_trace(terminal_state="completed")
        trace2 = self._make_trace(terminal_state="failed")

        diff = diff_traces(trace1, trace2)

        warnings = diff.regression_warnings
        assert any("regressed" in w.lower() for w in warnings)

    def test_summary_output(self):
        trace1 = self._make_trace(
            scene_id="scene_a",
            tool_calls=[("lookup_order", {})],
        )
        trace2 = self._make_trace(
            scene_id="scene_b",
            tool_calls=[("create_return", {})],
        )

        diff = diff_traces(trace1, trace2)
        summary = diff.summary()

        assert "scene_a" in summary
        assert "scene_b" in summary
        assert "Added" in summary or "Removed" in summary


class TestDiffToolSequences:
    def test_identical_sequences(self):
        seq1 = ["lookup_order", "create_return"]
        seq2 = ["lookup_order", "create_return"]

        result = diff_tool_sequences(seq1, seq2)

        assert result["similarity"] == 1.0

    def test_different_sequences(self):
        seq1 = ["lookup_order", "create_return"]
        seq2 = ["lookup_order", "cancel_order"]

        result = diff_tool_sequences(seq1, seq2)

        assert result["similarity"] < 1.0
        assert any(op["op"] == "replace" for op in result["operations"])

    def test_empty_sequences(self):
        result = diff_tool_sequences([], [])

        assert result["similarity"] == 1.0
        assert result["before_length"] == 0
        assert result["after_length"] == 0

    def test_sequence_with_additions(self):
        seq1 = ["lookup_order"]
        seq2 = ["lookup_order", "create_return", "send_email"]

        result = diff_tool_sequences(seq1, seq2)

        assert any(op["op"] == "insert" for op in result["operations"])
        assert result["after_length"] > result["before_length"]
