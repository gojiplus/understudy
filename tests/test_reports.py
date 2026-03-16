"""Tests for the ReportGenerator."""

import pytest

from understudy import Expectations, Persona, RunStorage, Scene, Trace, Turn, check
from understudy.reports import ReportGenerator


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
        storage.save(trace, scene, check_result=check_result)

    return storage


class TestReportGenerator:
    def test_generate_index(self, storage_with_runs):
        generator = ReportGenerator(storage_with_runs)
        html = generator.generate_index()
        assert "<html" in html
        assert "scene_0" in html or "scene_1" in html or "scene_2" in html

    def test_generate_run_report(self, storage_with_runs):
        generator = ReportGenerator(storage_with_runs)
        run_ids = storage_with_runs.list_runs()
        html = generator.generate_run_report(run_ids[0])
        assert "<html" in html
        assert "Run:" in html or "run" in html.lower()

    def test_generate_run_report_not_found(self, storage_with_runs):
        generator = ReportGenerator(storage_with_runs)
        with pytest.raises(FileNotFoundError):
            generator.generate_run_report("nonexistent_run")

    def test_generate_static_report(self, storage_with_runs, tmp_path):
        generator = ReportGenerator(storage_with_runs)
        output_dir = tmp_path / "output" / "report"
        generator.generate_static_report(output_dir)

        assert (output_dir / "index.html").exists()
        assert (output_dir / "runs").is_dir()

        index_content = (output_dir / "index.html").read_text()
        assert "<html" in index_content
        assert "runs/" in index_content

        run_ids = storage_with_runs.list_runs()
        for run_id in run_ids:
            run_file = output_dir / "runs" / f"{run_id}.html"
            assert run_file.exists()
            run_content = run_file.read_text()
            assert "<html" in run_content
            assert "../index.html" in run_content


class TestComparisonReport:
    @pytest.fixture
    def storage_with_tagged_runs(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for version in ["v1", "v2"]:
            for i in range(2):
                trace = Trace(
                    scene_id=f"scene_{i}",
                    turns=[Turn(role="user", content="hi"), Turn(role="agent", content="hello")],
                    terminal_state="done" if version == "v2" else "failed",
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

    def test_generate_comparison_report(self, storage_with_tagged_runs):
        generator = ReportGenerator(storage_with_tagged_runs)
        html = generator.generate_comparison_report(
            tag="version",
            before_value="v1",
            after_value="v2",
        )
        assert "<html" in html
        assert "v1" in html
        assert "v2" in html

    def test_generate_comparison_report_with_labels(self, storage_with_tagged_runs):
        generator = ReportGenerator(storage_with_tagged_runs)
        html = generator.generate_comparison_report(
            tag="version",
            before_value="v1",
            after_value="v2",
            before_label="Baseline",
            after_label="Candidate",
        )
        assert "Baseline" in html
        assert "Candidate" in html

    def test_generate_comparison_report_missing_tag(self, storage_with_tagged_runs):
        generator = ReportGenerator(storage_with_tagged_runs)
        with pytest.raises(ValueError) as exc_info:
            generator.generate_comparison_report(
                tag="version",
                before_value="v0",
                after_value="v1",
            )
        assert "No runs found" in str(exc_info.value)
