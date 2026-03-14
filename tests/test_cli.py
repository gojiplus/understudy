"""Tests for the CLI commands."""

import pytest
from click.testing import CliRunner

from understudy import Expectations, Persona, RunStorage, Scene, Trace, Turn, check
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
            expectations=Expectations(allowed_terminal_states=["done"]),
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
                    expectations=Expectations(allowed_terminal_states=["done"]),
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
