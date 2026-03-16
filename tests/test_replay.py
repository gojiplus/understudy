"""Tests for trace replay functionality."""

import json

from understudy import ReplayResult, ToolCall, Trace, Turn, load_trace, replay
from understudy.runner import AgentResponse


class MockAgentApp:
    """Mock agent app for replay testing."""

    def __init__(self, responses: list[AgentResponse] | None = None):
        self.responses = responses or []
        self.response_index = 0
        self.received_messages: list[str] = []

    def start(self, mocks=None):
        self.response_index = 0
        self.received_messages = []

    def send(self, message: str) -> AgentResponse:
        self.received_messages.append(message)
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return AgentResponse(content="Default response", terminal_state="completed")

    def stop(self):
        pass


class TestReplay:
    def _make_trace(
        self,
        user_messages: list[str],
        tool_calls: list[list[tuple[str, dict]]] | None = None,
    ) -> Trace:
        turns = []
        for i, msg in enumerate(user_messages):
            turns.append(Turn(role="user", content=msg))
            calls = []
            if tool_calls and i < len(tool_calls):
                calls = [ToolCall(tool_name=name, arguments=args) for name, args in tool_calls[i]]
            turns.append(Turn(role="agent", content=f"Response {i}", tool_calls=calls))

        return Trace(scene_id="test", turns=turns, terminal_state="completed")

    def test_replay_with_matching_behavior(self):
        original = self._make_trace(
            user_messages=["Hello", "How are you?"],
            tool_calls=[[("greet", {})], [("check_status", {})]],
        )

        app = MockAgentApp(
            [
                AgentResponse(content="Hi", tool_calls=[ToolCall(tool_name="greet", arguments={})]),
                AgentResponse(
                    content="I'm good",
                    tool_calls=[ToolCall(tool_name="check_status", arguments={})],
                    terminal_state="completed",
                ),
            ]
        )

        result = replay(original, app)

        assert result.match_rate == 1.0
        assert result.fully_matched
        assert result.diverged_at_turn is None

    def test_replay_with_divergent_behavior(self):
        original = self._make_trace(
            user_messages=["Hello", "Process order"],
            tool_calls=[[("greet", {})], [("process_order", {})]],
        )

        app = MockAgentApp(
            [
                AgentResponse(content="Hi", tool_calls=[ToolCall(tool_name="greet", arguments={})]),
                AgentResponse(
                    content="Error",
                    tool_calls=[ToolCall(tool_name="cancel_order", arguments={})],
                    terminal_state="failed",
                ),
            ]
        )

        result = replay(original, app)

        assert result.match_rate < 1.0
        assert not result.fully_matched

    def test_replay_sends_correct_messages(self):
        original = self._make_trace(user_messages=["Message 1", "Message 2", "Message 3"])

        app = MockAgentApp(
            [
                AgentResponse(content="Response 1"),
                AgentResponse(content="Response 2"),
                AgentResponse(content="Response 3", terminal_state="completed"),
            ]
        )
        replay(original, app)

        assert app.received_messages == ["Message 1", "Message 2", "Message 3"]

    def test_replay_result_summary(self):
        original = self._make_trace(user_messages=["Hello"])

        app = MockAgentApp([AgentResponse(content="Hi", terminal_state="completed")])

        result = replay(original, app)
        summary = result.summary()

        assert "Match rate" in summary
        assert "test" in summary


class TestLoadTrace:
    def test_load_trace_from_json(self, tmp_path):
        trace_data = {
            "scene_id": "test_scene",
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "agent", "content": "Hi there"},
            ],
            "terminal_state": "completed",
        }

        trace_file = tmp_path / "trace.json"
        trace_file.write_text(json.dumps(trace_data))

        trace = load_trace(trace_file)

        assert trace.scene_id == "test_scene"
        assert len(trace.turns) == 2
        assert trace.terminal_state == "completed"

    def test_load_trace_from_run_data(self, tmp_path):
        run_data = {
            "trace": {
                "scene_id": "nested_scene",
                "turns": [{"role": "user", "content": "Test"}],
            }
        }

        trace_file = tmp_path / "run.json"
        trace_file.write_text(json.dumps(run_data))

        trace = load_trace(trace_file)

        assert trace.scene_id == "nested_scene"


class TestReplayResult:
    def test_match_rate_calculation(self):
        trace = Trace(scene_id="test", turns=[])
        result = ReplayResult(
            original_trace=trace,
            new_trace=trace,
            matched_responses=3,
            total_turns=5,
        )

        assert result.match_rate == 0.6

    def test_match_rate_zero_turns(self):
        trace = Trace(scene_id="test", turns=[])
        result = ReplayResult(
            original_trace=trace,
            new_trace=trace,
            matched_responses=0,
            total_turns=0,
        )

        assert result.match_rate == 0.0

    def test_fully_matched_true(self):
        trace = Trace(scene_id="test", turns=[])
        result = ReplayResult(
            original_trace=trace,
            new_trace=trace,
            matched_responses=5,
            total_turns=5,
        )

        assert result.fully_matched

    def test_fully_matched_false(self):
        trace = Trace(scene_id="test", turns=[])
        result = ReplayResult(
            original_trace=trace,
            new_trace=trace,
            matched_responses=4,
            total_turns=5,
        )

        assert not result.fully_matched
