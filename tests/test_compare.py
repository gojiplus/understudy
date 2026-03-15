"""Tests for run comparison functionality."""

import pytest

from understudy import (
    ComparisonResult,
    Expectations,
    Persona,
    RunStorage,
    Scene,
    ToolCall,
    Trace,
    Turn,
    check,
    compare_runs,
)


def _make_trace(scene_id: str, tools: list[str], terminal_state: str = "done") -> Trace:
    """Helper to create a trace with given tools."""
    return Trace(
        scene_id=scene_id,
        turns=[
            Turn(
                role="agent",
                content="ok",
                tool_calls=[ToolCall(tool_name=t, arguments={}) for t in tools],
            )
        ],
        terminal_state=terminal_state,
    )


def _make_scene(scene_id: str) -> Scene:
    """Helper to create a minimal scene."""
    return Scene(
        id=scene_id,
        starting_prompt="hi",
        conversation_plan="test",
        persona=Persona(description="test"),
        expectations=Expectations(),
    )


class TestCompareRuns:
    def test_basic_comparison(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for i in range(2):
            trace = _make_trace(f"scene_{i}", ["lookup_order"])
            scene = _make_scene(f"scene_{i}")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"prompt": "v1"})

        for i in range(3):
            trace = _make_trace(f"scene_{i}", ["lookup_order", "create_return"])
            scene = _make_scene(f"scene_{i}")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"prompt": "v2"})

        result = compare_runs(storage, tag="prompt", before_value="v1", after_value="v2")

        assert isinstance(result, ComparisonResult)
        assert result.tag == "prompt"
        assert result.before_value == "v1"
        assert result.after_value == "v2"
        assert result.before_runs == 2
        assert result.after_runs == 3
        assert result.before_pass_rate == 1.0
        assert result.after_pass_rate == 1.0
        assert result.pass_rate_delta == 0.0

    def test_untagged_runs_excluded(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace1 = _make_trace("scene_tagged", ["tool_a"])
        scene1 = _make_scene("scene_tagged")
        check1 = check(trace1, scene1.expectations)
        storage.save(trace1, scene1, check_result=check1, tags={"model": "gpt-4"})

        trace2 = _make_trace("scene_untagged", ["tool_b"])
        scene2 = _make_scene("scene_untagged")
        check2 = check(trace2, scene2.expectations)
        storage.save(trace2, scene2, check_result=check2)

        trace3 = _make_trace("scene_tagged2", ["tool_c"])
        scene3 = _make_scene("scene_tagged2")
        check3 = check(trace3, scene3.expectations)
        storage.save(trace3, scene3, check_result=check3, tags={"model": "claude"})

        result = compare_runs(storage, tag="model", before_value="gpt-4", after_value="claude")

        assert result.before_runs == 1
        assert result.after_runs == 1

    def test_multi_tag_filter_by_specific(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace1 = _make_trace("scene_1", ["tool_a"])
        scene1 = _make_scene("scene_1")
        check1 = check(trace1, scene1.expectations)
        storage.save(trace1, scene1, check_result=check1, tags={"model": "gpt-4", "prompt": "v1"})

        trace2 = _make_trace("scene_2", ["tool_b"])
        scene2 = _make_scene("scene_2")
        check2 = check(trace2, scene2.expectations)
        storage.save(trace2, scene2, check_result=check2, tags={"model": "claude", "prompt": "v1"})

        trace3 = _make_trace("scene_3", ["tool_c"])
        scene3 = _make_scene("scene_3")
        check3 = check(trace3, scene3.expectations)
        storage.save(trace3, scene3, check_result=check3, tags={"model": "gpt-4", "prompt": "v2"})

        result = compare_runs(storage, tag="model", before_value="gpt-4", after_value="claude")

        assert result.before_runs == 2
        assert result.after_runs == 1

    def test_label_overrides(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace1 = _make_trace("scene_1", ["tool_a"])
        scene1 = _make_scene("scene_1")
        check1 = check(trace1, scene1.expectations)
        storage.save(trace1, scene1, check_result=check1, tags={"version": "1.0"})

        trace2 = _make_trace("scene_2", ["tool_b"])
        scene2 = _make_scene("scene_2")
        check2 = check(trace2, scene2.expectations)
        storage.save(trace2, scene2, check_result=check2, tags={"version": "2.0"})

        result = compare_runs(
            storage,
            tag="version",
            before_value="1.0",
            after_value="2.0",
            before_label="Baseline",
            after_label="Candidate",
        )

        assert result.before_label == "Baseline"
        assert result.after_label == "Candidate"
        assert result.before_value == "1.0"
        assert result.after_value == "2.0"

    def test_tool_usage_tracking(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace1 = _make_trace("scene_1", ["lookup_order", "get_policy"])
        scene1 = _make_scene("scene_1")
        check1 = check(trace1, scene1.expectations)
        storage.save(trace1, scene1, check_result=check1, tags={"version": "before"})

        trace2 = _make_trace("scene_2", ["lookup_order", "create_return", "send_email"])
        scene2 = _make_scene("scene_2")
        check2 = check(trace2, scene2.expectations)
        storage.save(trace2, scene2, check_result=check2, tags={"version": "after"})

        result = compare_runs(storage, tag="version", before_value="before", after_value="after")

        assert result.tool_usage_before["lookup_order"] == 1
        assert result.tool_usage_before["get_policy"] == 1
        assert "create_return" not in result.tool_usage_before

        assert result.tool_usage_after["lookup_order"] == 1
        assert result.tool_usage_after["create_return"] == 1
        assert result.tool_usage_after["send_email"] == 1

    def test_terminal_state_tracking(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for state in ["done", "done", "failed"]:
            trace = _make_trace("scene", ["tool_a"], terminal_state=state)
            scene = _make_scene("scene")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"version": "before"})

        for state in ["done", "failed", "failed"]:
            trace = _make_trace("scene", ["tool_a"], terminal_state=state)
            scene = _make_scene("scene")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"version": "after"})

        result = compare_runs(storage, tag="version", before_value="before", after_value="after")

        assert result.terminal_states_before["done"] == 2
        assert result.terminal_states_before["failed"] == 1
        assert result.terminal_states_after["done"] == 1
        assert result.terminal_states_after["failed"] == 2

    def test_empty_before_group_raises(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace = _make_trace("scene_1", ["tool_a"])
        scene = _make_scene("scene_1")
        check_result = check(trace, scene.expectations)
        storage.save(trace, scene, check_result=check_result, tags={"version": "after"})

        with pytest.raises(ValueError, match="No runs found with version=before"):
            compare_runs(storage, tag="version", before_value="before", after_value="after")

    def test_empty_after_group_raises(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace = _make_trace("scene_1", ["tool_a"])
        scene = _make_scene("scene_1")
        check_result = check(trace, scene.expectations)
        storage.save(trace, scene, check_result=check_result, tags={"version": "before"})

        with pytest.raises(ValueError, match="No runs found with version=after"):
            compare_runs(storage, tag="version", before_value="before", after_value="after")

    def test_missing_tag_key_no_match(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace = _make_trace("scene_1", ["tool_a"])
        scene = _make_scene("scene_1")
        check_result = check(trace, scene.expectations)
        storage.save(trace, scene, check_result=check_result, tags={"other_tag": "value"})

        with pytest.raises(ValueError, match="No runs found with nonexistent=value1"):
            compare_runs(storage, tag="nonexistent", before_value="value1", after_value="value2")

    def test_pass_rate_delta(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace1 = _make_trace("scene_1", ["tool_a"])
        scene1 = _make_scene("scene_1")
        check1 = check(trace1, scene1.expectations)
        storage.save(trace1, scene1, check_result=check1, tags={"version": "before"})

        trace2 = _make_trace("scene_2", ["bad_tool"])
        scene2 = _make_scene("scene_2")
        scene2.expectations = Expectations(forbidden_tools=["bad_tool"])
        check2 = check(trace2, scene2.expectations)
        storage.save(trace2, scene2, check_result=check2, tags={"version": "before"})

        trace3 = _make_trace("scene_3", ["tool_a"])
        scene3 = _make_scene("scene_3")
        check3 = check(trace3, scene3.expectations)
        storage.save(trace3, scene3, check_result=check3, tags={"version": "after"})

        trace4 = _make_trace("scene_4", ["tool_a"])
        scene4 = _make_scene("scene_4")
        check4 = check(trace4, scene4.expectations)
        storage.save(trace4, scene4, check_result=check4, tags={"version": "after"})

        result = compare_runs(storage, tag="version", before_value="before", after_value="after")

        assert result.before_pass_rate == 0.5
        assert result.after_pass_rate == 1.0
        assert result.pass_rate_delta == 0.5

    def test_avg_turns_delta(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for _ in range(2):
            trace = Trace(
                scene_id="scene",
                turns=[
                    Turn(role="user", content="hello"),
                    Turn(role="agent", content="hi"),
                ],
                terminal_state="done",
            )
            scene = _make_scene("scene")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"version": "before"})

        for _ in range(2):
            trace = Trace(
                scene_id="scene",
                turns=[
                    Turn(role="user", content="hello"),
                    Turn(role="agent", content="hi"),
                    Turn(role="user", content="thanks"),
                    Turn(role="agent", content="bye"),
                ],
                terminal_state="done",
            )
            scene = _make_scene("scene")
            check_result = check(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result, tags={"version": "after"})

        result = compare_runs(storage, tag="version", before_value="before", after_value="after")

        assert result.before_avg_turns == 2.0
        assert result.after_avg_turns == 4.0
        assert result.avg_turns_delta == 2.0
