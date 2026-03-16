"""Tests for trajectory matching metric."""

from understudy import Expectations, ToolCall, Trace, Turn, check
from understudy.metrics.builtins import _compute_trajectory_score, compute_trajectory_match


class TestTrajectoryScore:
    def test_exact_match(self):
        score, _ = _compute_trajectory_score(["a", "b", "c"], ["a", "b", "c"], "exact")
        assert score == 1.0

    def test_exact_mismatch_different_order(self):
        score, _ = _compute_trajectory_score(["a", "c", "b"], ["a", "b", "c"], "exact")
        assert score == 0.0

    def test_exact_mismatch_different_length(self):
        score, _ = _compute_trajectory_score(["a", "b"], ["a", "b", "c"], "exact")
        assert score == 0.0

    def test_exact_mismatch_extra_tools(self):
        score, _ = _compute_trajectory_score(["a", "b", "c", "d"], ["a", "b", "c"], "exact")
        assert score == 0.0

    def test_prefix_match(self):
        score, _ = _compute_trajectory_score(["a", "b", "c", "d"], ["a", "b"], "prefix")
        assert score == 1.0

    def test_prefix_match_exact(self):
        score, _ = _compute_trajectory_score(["a", "b"], ["a", "b"], "prefix")
        assert score == 1.0

    def test_prefix_mismatch(self):
        score, _ = _compute_trajectory_score(["x", "b", "c"], ["a", "b"], "prefix")
        assert score == 0.0

    def test_prefix_mismatch_too_short(self):
        score, _ = _compute_trajectory_score(["a"], ["a", "b"], "prefix")
        assert score == 0.0

    def test_contains_match(self):
        score, _ = _compute_trajectory_score(["a", "x", "b", "y", "c"], ["a", "b", "c"], "contains")
        assert score == 1.0

    def test_contains_match_exact(self):
        score, _ = _compute_trajectory_score(["a", "b", "c"], ["a", "b", "c"], "contains")
        assert score == 1.0

    def test_contains_match_subset(self):
        score, _ = _compute_trajectory_score(["a", "b", "c", "d"], ["a", "c"], "contains")
        assert score == 1.0

    def test_contains_mismatch_wrong_order(self):
        score, _ = _compute_trajectory_score(["c", "b", "a"], ["a", "b", "c"], "contains")
        assert score == 0.0

    def test_contains_mismatch_missing(self):
        score, _ = _compute_trajectory_score(["a", "b"], ["a", "b", "c"], "contains")
        assert score == 0.0

    def test_subset_match(self):
        score, _ = _compute_trajectory_score(["c", "a", "b"], ["a", "c"], "subset")
        assert score == 1.0

    def test_subset_match_any_order(self):
        score, _ = _compute_trajectory_score(["c", "b", "a"], ["a", "b", "c"], "subset")
        assert score == 1.0

    def test_subset_match_with_extras(self):
        score, _ = _compute_trajectory_score(["a", "x", "b", "y"], ["a", "b"], "subset")
        assert score == 1.0

    def test_subset_mismatch_missing(self):
        score, detail = _compute_trajectory_score(["a", "b"], ["a", "b", "c"], "subset")
        assert score == 0.0
        assert "missing" in detail
        assert "c" in detail

    def test_unknown_mode(self):
        score, detail = _compute_trajectory_score(["a"], ["a"], "unknown_mode")
        assert score == 0.0
        assert "Unknown mode" in detail


class TestTrajectoryMatchMetric:
    def _make_trace(self, tool_names: list[str]) -> Trace:
        return Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="done",
                    tool_calls=[ToolCall(tool_name=name, arguments={}) for name in tool_names],
                )
            ],
        )

    def test_no_expected_trajectory(self):
        trace = self._make_trace(["a", "b", "c"])
        expectations = Expectations()
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is None
        assert "No expected trajectory" in result.detail

    def test_exact_match_pass(self):
        trace = self._make_trace(["lookup_order", "get_return_policy", "create_return"])
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy", "create_return"],
            trajectory_match_mode="exact",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is True
        assert result.value["score"] == 1.0
        assert result.value["mode"] == "exact"

    def test_exact_match_fail(self):
        trace = self._make_trace(["lookup_order", "create_return"])
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy", "create_return"],
            trajectory_match_mode="exact",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is False
        assert result.value["score"] == 0.0

    def test_contains_mode(self):
        trace = self._make_trace(["lookup_order", "extra_tool", "get_return_policy"])
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy"],
            trajectory_match_mode="contains",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is True

    def test_subset_mode(self):
        trace = self._make_trace(["get_return_policy", "lookup_order"])
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy"],
            trajectory_match_mode="subset",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is True


class TestTrajectoryMatchWithCheck:
    def test_check_with_trajectory_match_metric(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="done",
                    tool_calls=[
                        ToolCall(tool_name="lookup_order", arguments={}),
                        ToolCall(tool_name="get_return_policy", arguments={}),
                    ],
                )
            ],
        )
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy"],
            trajectory_match_mode="exact",
            metrics=["trajectory_match"],
        )
        result = check(trace, expectations)
        assert "trajectory_match" in result.metrics
        assert result.metrics["trajectory_match"].passed is True

    def test_check_trajectory_mismatch_in_metrics(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="done",
                    tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                )
            ],
        )
        expectations = Expectations(
            expected_trajectory=["lookup_order", "get_return_policy"],
            trajectory_match_mode="exact",
            metrics=["trajectory_match"],
        )
        result = check(trace, expectations)
        assert result.metrics["trajectory_match"].passed is False


class TestEdgeCases:
    def test_empty_actual_trajectory(self):
        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="done", tool_calls=[])],
        )
        expectations = Expectations(
            expected_trajectory=["lookup_order"],
            trajectory_match_mode="exact",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is False

    def test_empty_expected_trajectory(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="done",
                    tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                )
            ],
        )
        expectations = Expectations(
            expected_trajectory=[],
            trajectory_match_mode="exact",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is False

    def test_both_empty(self):
        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="done", tool_calls=[])],
        )
        expectations = Expectations(
            expected_trajectory=[],
            trajectory_match_mode="exact",
        )
        result = compute_trajectory_match(trace, expectations)
        assert result.passed is True

    def test_default_mode_is_exact(self):
        expectations = Expectations(
            expected_trajectory=["lookup_order"],
        )
        assert expectations.trajectory_match_mode == "exact"
