"""Tests for the CLI commands."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from understudy import (
    Expectations,
    Persona,
    RunStorage,
    Scene,
    Trace,
    TraceStorage,
    Turn,
    check,
)
from understudy.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def storage_with_runs(tmp_path):
    storage = RunStorage(path=tmp_path / "runs")

    for i in range(3):
        trace = Trace(
            scene_id=f"scene_{i}",
            turns=[Turn(role="user", content="hello"), Turn(role="agent", content="hi")],
            terminal_state="done" if i < 2 else "failed",
        )
        scene = Scene(
            id=f"scene_{i}",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
            expectations=Expectations(),
        )
        check_result = check(trace, scene.expectations)
        tags = {"version": "v1"} if i < 2 else {"version": "v2"}
        storage.save(trace, scene, check_result=check_result, tags=tags)

    return storage


class TestListCommand:
    def test_list_no_runs(self, runner, tmp_path):
        runs_path = tmp_path / "empty_runs"
        runs_path.mkdir()
        result = runner.invoke(main, ["list", "--runs", str(runs_path)])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_list_with_runs(self, runner, storage_with_runs):
        result = runner.invoke(main, ["list", "--runs", str(storage_with_runs.path)])
        assert result.exit_code == 0
        assert "Found 3 runs" in result.output
        assert "[PASS]" in result.output or "[FAIL]" in result.output


class TestSummaryCommand:
    def test_summary_no_runs(self, runner, tmp_path):
        runs_path = tmp_path / "empty_runs"
        runs_path.mkdir()
        result = runner.invoke(main, ["summary", "--runs", str(runs_path)])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_summary_with_runs(self, runner, storage_with_runs):
        result = runner.invoke(main, ["summary", "--runs", str(storage_with_runs.path)])
        assert result.exit_code == 0
        assert "understudy Summary" in result.output
        assert "Total Runs:" in result.output
        assert "Pass Rate:" in result.output


class TestShowCommand:
    def test_show_not_found(self, runner, tmp_path):
        runs_path = tmp_path / "empty_runs"
        runs_path.mkdir()
        result = runner.invoke(main, ["show", "nonexistent", "--runs", str(runs_path)])
        assert result.exit_code == 1
        assert "Run not found" in result.output

    def test_show_existing_run(self, runner, storage_with_runs):
        run_ids = storage_with_runs.list_runs()
        result = runner.invoke(main, ["show", run_ids[0], "--runs", str(storage_with_runs.path)])
        assert result.exit_code == 0
        assert "Run:" in result.output
        assert "Scene:" in result.output
        assert "Status:" in result.output


class TestDeleteCommand:
    def test_delete_not_found(self, runner, tmp_path):
        runs_path = tmp_path / "empty_runs"
        runs_path.mkdir()
        result = runner.invoke(main, ["delete", "nonexistent", "--runs", str(runs_path)])
        assert result.exit_code == 1
        assert "Run not found" in result.output

    def test_delete_with_confirm(self, runner, storage_with_runs):
        run_ids = storage_with_runs.list_runs()
        run_id = run_ids[0]
        result = runner.invoke(
            main, ["delete", run_id, "--runs", str(storage_with_runs.path)], input="y\n"
        )
        assert result.exit_code == 0
        assert "Deleted:" in result.output
        assert len(storage_with_runs.list_runs()) == 2

    def test_delete_with_yes_flag(self, runner, storage_with_runs):
        run_ids = storage_with_runs.list_runs()
        run_id = run_ids[0]
        result = runner.invoke(
            main, ["delete", run_id, "--yes", "--runs", str(storage_with_runs.path)]
        )
        assert result.exit_code == 0
        assert "Deleted:" in result.output


class TestClearCommand:
    def test_clear_no_runs(self, runner, tmp_path):
        runs_path = tmp_path / "empty_runs"
        runs_path.mkdir()
        result = runner.invoke(main, ["clear", "--runs", str(runs_path)])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_clear_with_confirm(self, runner, storage_with_runs):
        result = runner.invoke(main, ["clear", "--runs", str(storage_with_runs.path)], input="y\n")
        assert result.exit_code == 0
        assert "Cleared 3 runs" in result.output
        assert len(storage_with_runs.list_runs()) == 0

    def test_clear_with_yes_flag(self, runner, storage_with_runs):
        result = runner.invoke(main, ["clear", "--yes", "--runs", str(storage_with_runs.path)])
        assert result.exit_code == 0
        assert "Cleared 3 runs" in result.output


class TestCompareCommand:
    @pytest.fixture
    def storage_with_tagged_runs(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for version, passed_count in [("v1", 2), ("v2", 3)]:
            for i in range(3):
                trace = Trace(
                    scene_id=f"scene_{i}",
                    turns=[Turn(role="user", content="hi"), Turn(role="agent", content="hello")],
                    terminal_state="done" if i < passed_count else "failed",
                )
                scene = Scene(
                    id=f"scene_{i}",
                    starting_prompt="hi",
                    conversation_plan="test",
                    persona=Persona(description="test"),
                    expectations=Expectations(),
                )
                check_result = check(trace, scene.expectations)
                storage.save(trace, scene, check_result=check_result, tags={"version": version})

        return storage

    def test_compare_missing_tag(self, runner, storage_with_tagged_runs):
        result = runner.invoke(
            main,
            [
                "compare",
                "--runs",
                str(storage_with_tagged_runs.path),
                "--tag",
                "version",
                "--before",
                "v0",
                "--after",
                "v1",
            ],
        )
        assert result.exit_code == 1
        assert "No runs found" in result.output

    def test_compare_success(self, runner, storage_with_tagged_runs):
        result = runner.invoke(
            main,
            [
                "compare",
                "--runs",
                str(storage_with_tagged_runs.path),
                "--tag",
                "version",
                "--before",
                "v1",
                "--after",
                "v2",
            ],
        )
        assert result.exit_code == 0
        assert "Comparison:" in result.output
        assert "Pass Rate" in result.output

    def test_compare_html_output(self, runner, storage_with_tagged_runs, tmp_path):
        output_file = tmp_path / "comparison.html"
        result = runner.invoke(
            main,
            [
                "compare",
                "--runs",
                str(storage_with_tagged_runs.path),
                "--tag",
                "version",
                "--before",
                "v1",
                "--after",
                "v2",
                "--html",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Comparison report generated" in result.output


class TestRunCommand:
    @pytest.fixture(autouse=True)
    def mock_litellm_for_run(self, monkeypatch):
        """Mock litellm.completion for all run command tests."""

        def mock_completion(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="<finished>"))]
            return response

        monkeypatch.setattr("litellm.completion", mock_completion)

    @pytest.fixture
    def scene_file(self, tmp_path):
        scene_content = """\
id: test_scene
description: Test scene for CLI
starting_prompt: Hello, I need help
conversation_plan: Ask for help and accept the response
persona: cooperative
expectations:
  required_tools: []
"""
        scene_path = tmp_path / "test_scene.yaml"
        scene_path.write_text(scene_content)
        return scene_path

    @pytest.fixture
    def scene_dir(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        for i in range(2):
            scene_content = f"""\
id: scene_{i}
description: Test scene {i}
starting_prompt: Hello {i}
conversation_plan: Simple test
persona: cooperative
expectations:
  required_tools: []
"""
            (scenes_dir / f"scene_{i}.yaml").write_text(scene_content)

        return scenes_dir

    @pytest.fixture
    def mock_app_module(self, tmp_path):
        module_content = """\
from understudy.runner import AgentResponse

class MockApp:
    def start(self, mocks=None):
        pass

    def send(self, message):
        return AgentResponse(
            content="I can help with that!",
            terminal_state="done",
        )

    def stop(self):
        pass

app = MockApp()
"""
        module_path = tmp_path / "mock_app.py"
        module_path.write_text(module_content)
        return tmp_path

    @pytest.fixture
    def mock_mocks_module(self, tmp_path):
        module_content = """\
from understudy.mocks import MockToolkit

def create_mocks():
    toolkit = MockToolkit()

    @toolkit.handle("test_tool")
    def test_tool():
        return {"result": "ok"}

    return toolkit
"""
        module_path = tmp_path / "mock_mocks.py"
        module_path.write_text(module_content)
        return tmp_path

    def test_run_invalid_import_path(self, runner, scene_file):
        result = runner.invoke(
            main,
            [
                "run",
                "--app",
                "invalid_path_no_colon",
                "--scene",
                str(scene_file),
            ],
        )
        assert result.exit_code != 0
        assert "Invalid import path" in result.output

    def test_run_module_not_found(self, runner, scene_file):
        result = runner.invoke(
            main,
            [
                "run",
                "--app",
                "nonexistent_module:app",
                "--scene",
                str(scene_file),
            ],
        )
        assert result.exit_code != 0
        assert "Cannot import module" in result.output

    def test_run_single_scene(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--runs",
                    str(runs_path),
                ],
            )
            assert "Loaded scene: test_scene" in result.output
            assert "Running with simulator model: gpt-4o" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_scene_directory(self, runner, scene_dir, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_dir),
                    "--runs",
                    str(runs_path),
                ],
            )
            assert "Loaded 2 scenes from" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_with_custom_model(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--simulator-model",
                    "claude-sonnet-4-20250514",
                    "--runs",
                    str(runs_path),
                ],
            )
            assert "Running with simulator model: claude-sonnet-4-20250514" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_with_tags(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--tag",
                    "version=v1",
                    "--tag",
                    "model=test",
                    "--runs",
                    str(runs_path),
                ],
            )
            assert result.exit_code == 0
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_invalid_tag_format(self, runner, scene_file, mock_app_module):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--tag",
                    "invalid_tag_no_equals",
                ],
            )
            assert result.exit_code != 0
            assert "Invalid tag format" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_with_mocks(self, runner, scene_file, mock_app_module, mock_mocks_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        sys.path.insert(0, str(mock_mocks_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--mocks",
                    "mock_mocks:create_mocks",
                    "--runs",
                    str(runs_path),
                ],
            )
            assert "Using mocks:" in result.output
            assert "test_tool" in result.output
        finally:
            sys.path.remove(str(mock_app_module))
            sys.path.remove(str(mock_mocks_module))

    def test_run_with_junit_output(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            junit_path = tmp_path / "results.xml"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--runs",
                    str(runs_path),
                    "--junit",
                    str(junit_path),
                ],
            )
            assert "JUnit XML exported to:" in result.output
            assert junit_path.exists()
        finally:
            sys.path.remove(str(mock_app_module))

    def test_run_with_n_sims(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            runs_path = tmp_path / "runs"
            result = runner.invoke(
                main,
                [
                    "run",
                    "--app",
                    "mock_app:app",
                    "--scene",
                    str(scene_file),
                    "--runs",
                    str(runs_path),
                    "--n-sims",
                    "2",
                ],
            )
            assert "Simulations per scene: 2" in result.output
        finally:
            sys.path.remove(str(mock_app_module))


class TestSimulateCommand:
    @pytest.fixture(autouse=True)
    def mock_litellm_for_simulate(self, monkeypatch):
        """Mock litellm.completion for all simulate command tests."""

        def mock_completion(*args, **kwargs):
            response = MagicMock()
            response.choices = [MagicMock(message=MagicMock(content="<finished>"))]
            return response

        monkeypatch.setattr("litellm.completion", mock_completion)

    @pytest.fixture
    def scene_file(self, tmp_path):
        scene_content = """\
id: test_scene
description: Test scene for CLI
starting_prompt: Hello, I need help
conversation_plan: Ask for help and accept the response
persona: cooperative
expectations:
  required_tools: []
"""
        scene_path = tmp_path / "test_scene.yaml"
        scene_path.write_text(scene_content)
        return scene_path

    @pytest.fixture
    def scene_dir(self, tmp_path):
        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        for i in range(2):
            scene_content = f"""\
id: scene_{i}
description: Test scene {i}
starting_prompt: Hello {i}
conversation_plan: Simple test
persona: cooperative
expectations:
  required_tools: []
"""
            (scenes_dir / f"scene_{i}.yaml").write_text(scene_content)

        return scenes_dir

    @pytest.fixture
    def mock_app_module(self, tmp_path):
        module_content = """\
from understudy.runner import AgentResponse

class MockApp:
    def start(self, mocks=None):
        pass

    def send(self, message):
        return AgentResponse(
            content="I can help with that!",
            terminal_state="done",
        )

    def stop(self):
        pass

app = MockApp()
"""
        module_path = tmp_path / "mock_app.py"
        module_path.write_text(module_content)
        return tmp_path

    def test_simulate_single_scene(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            output_path = tmp_path / "traces"
            result = runner.invoke(
                main,
                [
                    "simulate",
                    "--app",
                    "mock_app:app",
                    "--scenes",
                    str(scene_file),
                    "--output",
                    str(output_path),
                ],
            )
            assert "Running simulations with model: gpt-4o" in result.output
            assert "Traces saved to:" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_simulate_with_n_sims(self, runner, scene_file, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            output_path = tmp_path / "traces"
            result = runner.invoke(
                main,
                [
                    "simulate",
                    "--app",
                    "mock_app:app",
                    "--scenes",
                    str(scene_file),
                    "--output",
                    str(output_path),
                    "--n-sims",
                    "3",
                ],
            )
            assert "1 scenes x 3 runs = 3 traces" in result.output
        finally:
            sys.path.remove(str(mock_app_module))

    def test_simulate_directory(self, runner, scene_dir, mock_app_module, tmp_path):
        import sys

        sys.path.insert(0, str(mock_app_module))
        try:
            output_path = tmp_path / "traces"
            result = runner.invoke(
                main,
                [
                    "simulate",
                    "--app",
                    "mock_app:app",
                    "--scenes",
                    str(scene_dir),
                    "--output",
                    str(output_path),
                    "--n-sims",
                    "2",
                ],
            )
            assert "2 scenes x 2 runs = 4 traces" in result.output
        finally:
            sys.path.remove(str(mock_app_module))


class TestEvaluateCommand:
    @pytest.fixture
    def trace_storage_with_traces(self, tmp_path):
        storage = TraceStorage(path=tmp_path / "traces")

        for i in range(2):
            trace = Trace(
                scene_id=f"scene_{i}",
                turns=[Turn(role="user", content="hello"), Turn(role="agent", content="hi")],
                terminal_state="completed",
            )
            scene = Scene(
                id=f"scene_{i}",
                starting_prompt="hello",
                conversation_plan="greet",
                persona=Persona(description="friendly"),
                expectations=Expectations(expected_resolution="completed"),
            )
            storage.save(trace, scene, sim_index=0)

        return storage

    def test_evaluate_traces(self, runner, trace_storage_with_traces, tmp_path):
        output_path = tmp_path / "results"
        result = runner.invoke(
            main,
            [
                "evaluate",
                "--traces",
                str(trace_storage_with_traces.path),
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        assert "Evaluating traces from:" in result.output
        assert "2/2 passed" in result.output
        assert "Results saved to:" in result.output

    def test_evaluate_with_junit(self, runner, trace_storage_with_traces, tmp_path):
        output_path = tmp_path / "results"
        junit_path = tmp_path / "results.xml"
        result = runner.invoke(
            main,
            [
                "evaluate",
                "--traces",
                str(trace_storage_with_traces.path),
                "--output",
                str(output_path),
                "--junit",
                str(junit_path),
            ],
        )
        assert result.exit_code == 0
        assert junit_path.exists()
        assert "JUnit XML exported to:" in result.output
