"""Example tests for agentic flows using understudy.

These tests demonstrate the agentic evaluation workflow:
1. Load an agentic scene from YAML
2. Run it against the mock agent with mocks
3. Assert against the agentic trace

To run:
    cd examples/agentic
    pytest test_agentic.py -v
"""

import pytest

from understudy.agentic import AgenticScene, check_agentic, run_agentic


def test_code_review_task(app, mocks, scenes_dir):
    """Agent must read file, analyze, and produce report without modifying."""
    scene = AgenticScene.from_file(scenes_dir / "code_review_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)

    assert trace.performed("read_file")
    assert trace.performed("analyze_code")
    assert trace.performed("write_report")
    assert not trace.performed("write_file")
    assert not trace.performed("delete_file")
    assert trace.outcome == "success"


def test_file_search_task(app, mocks, scenes_dir):
    """Agent must search directories and find matching files."""
    scene = AgenticScene.from_file(scenes_dir / "file_search_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)

    assert trace.performed("list_directory")
    assert trace.performed("search_content")
    assert not trace.performed("write_file")
    assert trace.outcome == "success"
    assert "results" in trace.final_state


def test_data_analysis_task(app, mocks, scenes_dir):
    """Agent must read data, compute stats, and output results."""
    scene = AgenticScene.from_file(scenes_dir / "data_analysis_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)

    assert trace.performed("read_file")
    assert trace.performed("compute_stats")
    assert trace.performed("write_output")
    assert trace.outcome == "success"


def test_code_review_check(app, mocks, scenes_dir):
    """Test using check_agentic for bulk validation."""
    scene = AgenticScene.from_file(scenes_dir / "code_review_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)
    result = check_agentic(trace, scene.expectations)

    assert result.passed, f"Failed checks:\n{result.summary()}"


def test_file_search_check(app, mocks, scenes_dir):
    """Test file search with check_agentic."""
    scene = AgenticScene.from_file(scenes_dir / "file_search_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)
    result = check_agentic(trace, scene.expectations)

    assert result.passed, f"Failed checks:\n{result.summary()}"


def test_data_analysis_check(app, mocks, scenes_dir):
    """Test data analysis with check_agentic."""
    scene = AgenticScene.from_file(scenes_dir / "data_analysis_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)
    result = check_agentic(trace, scene.expectations)

    assert result.passed, f"Failed checks:\n{result.summary()}"


@pytest.mark.parametrize(
    "scene_file",
    [
        "code_review_task.yaml",
        "file_search_task.yaml",
        "data_analysis_task.yaml",
    ],
)
def test_all_scenes_pass_expectations(app, mocks, scenes_dir, scene_file):
    """Parametrized test running all scenes."""
    scene = AgenticScene.from_file(scenes_dir / scene_file)
    trace = run_agentic(app, scene, mocks=mocks)
    result = check_agentic(trace, scene.expectations)

    assert result.passed, f"{scene.id} failed:\n{result.summary()}"


def test_trace_properties(app, mocks, scenes_dir):
    """Test that trace captures expected properties."""
    scene = AgenticScene.from_file(scenes_dir / "code_review_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)

    assert trace.total_steps > 0
    assert trace.total_tokens > 0
    assert trace.total_latency_ms >= 0
    assert trace.started_at is not None
    assert trace.finished_at is not None
    assert len(trace.action_sequence()) >= 3
    assert len(trace.thinking_steps()) >= 2


def test_reasoning_steps_captured(app, mocks, scenes_dir):
    """Test that reasoning/thinking steps are captured."""
    scene = AgenticScene.from_file(scenes_dir / "code_review_task.yaml")
    trace = run_agentic(app, scene, mocks=mocks)

    thinking = trace.thinking_steps()
    assert len(thinking) > 0
    assert all(step.reasoning is not None for step in thinking)
