"""Exercise turn and prefix commands using saved fixtures and assertions."""

import json

from click.testing import CliRunner
from test_execution import CounterApp

from understudy.cli import main


def test_turn_command_and_assertion_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("understudy.cli.import_object", lambda _: CounterApp)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"state": {"n": 4}}))
    expected = tmp_path / "expectations.json"
    expected.write_text(json.dumps({"required_tools": ["new"]}))
    output = tmp_path / "result.json"
    saved = tmp_path / "checkpoint.json"
    args = [
        "run-turn",
        "--factory",
        "app:factory",
        "--message",
        "next",
        "--fixture",
        str(fixture),
        "--expectations",
        str(expected),
        "--output",
        str(output),
        "--checkpoint-output",
        str(saved),
        "--runs",
        str(tmp_path / "runs"),
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["state_after"] == {"n": 5}
    assert saved.exists()
    expected.write_text(json.dumps({"forbidden_tools": ["new"]}))
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 1
    assert "Next-turn assertions failed" in result.output
    from understudy.reports import ReportGenerator
    from understudy.storage import RunStorage

    storage = RunStorage(tmp_path / "runs")
    report = ReportGenerator(storage).generate_run_report(storage.list_runs()[0])
    assert "Next-turn test" in report


def test_replay_command(tmp_path, monkeypatch):
    monkeypatch.setattr("understudy.cli.import_object", lambda _: CounterApp)
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "scene_id": "recorded",
                "turns": [
                    {"role": "user", "content": "one"},
                    {
                        "role": "agent",
                        "content": "1",
                        "tool_calls": [{"tool_name": "new"}],
                    },
                ],
            }
        )
    )
    checkpoint = tmp_path / "checkpoint.json"
    result = CliRunner().invoke(
        main,
        [
            "replay-until",
            "--factory",
            "app:factory",
            "--trace",
            str(trace),
            "--after-turn",
            "1",
            "--output",
            str(checkpoint),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(checkpoint.read_text())["after_turn"] == 1
    result = CliRunner().invoke(
        main,
        [
            "replay-until",
            "--factory",
            "app:factory",
            "--trace",
            str(trace),
            "--after-turn",
            "2",
            "--output",
            str(checkpoint),
        ],
    )
    assert result.exit_code == 1
    assert "outside" in result.output


def test_parallel_cli_creates_independent_mock_state(tmp_path, monkeypatch):
    from understudy import MockToolkit
    from understudy.runner import AgentResponse
    from understudy.storage import RunStorage, TraceStorage

    class EndingApp(CounterApp):
        def send(self, message):
            self.mocks.state["n"] = self.mocks.state.get("n", 0) + 1
            return AgentResponse(str(self.mocks.state["n"]), terminal_state="done")

    monkeypatch.setattr(
        "understudy.cli.import_object",
        lambda path: EndingApp if path == "app:factory" else MockToolkit,
    )
    scene = tmp_path / "scene.yaml"
    scene.write_text(
        "id: parallel\nstarting_prompt: hello\nconversation_plan: test\n"
        "persona:\n  description: test\n"
    )
    for command, scene_flag, output_flag in [
        ("run", "--scene", "--runs"),
        ("simulate", "--scenes", "--output"),
    ]:
        output = tmp_path / command
        result = CliRunner().invoke(
            main,
            [
                command,
                "--app",
                "app:factory",
                "--mocks",
                "app:mocks",
                scene_flag,
                str(scene),
                output_flag,
                str(output),
                "--parallel",
                "2",
                "--n-sims",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        storage = RunStorage(output) if command == "run" else TraceStorage(output)
        runs = storage.load_all()
        assert len(runs) == 4
        assert all(run["trace"].turns[-1].content == "1" for run in runs)
