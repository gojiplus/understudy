"""Tests for Judge and FailureAnalyzer."""

from unittest.mock import MagicMock, patch

import pytest

from understudy import Expectations, ToolCall, Trace, Turn
from understudy.judges import (
    FAILURE_ANALYSIS_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    FailureAnalysis,
    FailureAnalyzer,
    Judge,
    JudgeResult,
)


class TestJudgeResult:
    def test_unanimous_true(self):
        result = JudgeResult(score=1, raw_scores=[1, 1, 1, 1, 1], agreement_rate=1.0)
        assert result.unanimous is True

    def test_unanimous_false(self):
        result = JudgeResult(score=1, raw_scores=[1, 1, 1, 0, 0], agreement_rate=0.6)
        assert result.unanimous is False

    def test_agreement_rate_calculation(self):
        result = JudgeResult(score=0, raw_scores=[0, 0, 0, 1, 1], agreement_rate=0.6)
        assert result.score == 0
        assert result.agreement_rate == 0.6


class TestJudge:
    def _make_trace(self) -> Trace:
        return Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="I want to return my order"),
                Turn(
                    role="agent",
                    content="I'll help you with that",
                    tool_calls=[
                        ToolCall(tool_name="lookup_order", arguments={"id": "123"}),
                    ],
                ),
            ],
            terminal_state="completed",
        )

    def test_judge_init(self):
        judge = Judge(rubric="The agent was helpful.", samples=3, model="gpt-4o-mini")
        assert judge.rubric == "The agent was helpful."
        assert judge.samples == 3
        assert judge.model == "gpt-4o-mini"

    def test_judge_default_samples(self):
        judge = Judge(rubric="Test rubric")
        assert judge.samples == 5
        assert judge.model == "gpt-4o"

    def test_judge_evaluate_all_yes(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="YES"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="The agent was helpful.", samples=3)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 1
            assert result.raw_scores == [1, 1, 1]
            assert result.agreement_rate == 1.0
            assert mock_litellm.completion.call_count == 3

    def test_judge_evaluate_all_no(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="NO"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="The agent was rude.", samples=3)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 0
            assert result.raw_scores == [0, 0, 0]
            assert result.agreement_rate == 1.0

    def test_judge_evaluate_mixed_votes(self):
        mock_litellm = MagicMock()
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="YES"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="YES"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="YES"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="NO"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="NO"))]),
        ]
        mock_litellm.completion.side_effect = responses

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test rubric", samples=5)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 1
            assert result.raw_scores == [1, 1, 1, 0, 0]
            assert result.agreement_rate == 0.6

    def test_judge_evaluate_majority_no(self):
        mock_litellm = MagicMock()
        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content="NO"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="NO"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="NO"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="YES"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="YES"))]),
        ]
        mock_litellm.completion.side_effect = responses

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test rubric", samples=5)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 0
            assert result.raw_scores == [0, 0, 0, 1, 1]
            assert result.agreement_rate == 0.6

    def test_judge_handles_none_content(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test rubric", samples=1)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 0

    def test_judge_handles_lowercase_yes(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="yes"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test rubric", samples=1)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 1

    def test_judge_handles_yes_with_extra_text(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="YES, definitely"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test rubric", samples=1)
            trace = self._make_trace()
            result = judge.evaluate(trace)

            assert result.score == 1

    def test_judge_uses_correct_model(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="YES"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            judge = Judge(rubric="Test", samples=1, model="claude-3-haiku-20240307")
            trace = self._make_trace()
            judge.evaluate(trace)

            call_args = mock_litellm.completion.call_args
            assert call_args.kwargs["model"] == "claude-3-haiku-20240307"

    def test_judge_without_litellm_import_error(self):
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'litellm'")

        with (
            patch.dict("sys.modules", {"litellm": None}),
            patch("builtins.__import__", side_effect=raise_import_error),
        ):
            judge = Judge(rubric="Test", samples=1)
            trace = self._make_trace()
            with pytest.raises(ImportError, match="litellm"):
                judge.evaluate(trace)


class TestFailureAnalysis:
    def test_failure_analysis_dataclass(self):
        analysis = FailureAnalysis(
            run_id="run_123",
            scene_id="scene_456",
            failed_checks=["required_tool", "expected_resolution"],
            analysis="The agent failed to call the required tool.",
        )
        assert analysis.run_id == "run_123"
        assert analysis.scene_id == "scene_456"
        assert len(analysis.failed_checks) == 2
        assert "required tool" in analysis.analysis.lower()


class TestFailureAnalyzer:
    def _make_trace(self) -> Trace:
        return Trace(
            scene_id="test_scene",
            turns=[
                Turn(role="user", content="I need help"),
                Turn(role="agent", content="Sorry, I can't help"),
            ],
            terminal_state="failed",
        )

    def _make_scene(self):
        from understudy.models import Persona, Scene

        return Scene(
            id="test_scene",
            starting_prompt="I need help",
            conversation_plan="Help the user",
            persona=Persona(description="Cooperative"),
            expectations=Expectations(
                required_tools=["lookup_order"],
                expected_resolution="completed",
            ),
        )

    def test_analyzer_init(self):
        analyzer = FailureAnalyzer(model="gpt-4o-mini")
        assert analyzer.model == "gpt-4o-mini"

    def test_analyzer_default_model(self):
        analyzer = FailureAnalyzer()
        assert analyzer.model == "gpt-4o"

    def test_analyze_returns_analysis(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="The agent failed because it didn't look up the order.")
            )
        ]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            analyzer = FailureAnalyzer()
            trace = self._make_trace()
            result = analyzer.analyze(
                trace, "Required tools: lookup_order", ["required_tool: lookup_order NOT called"]
            )

            assert "failed" in result.lower()

    def test_analyze_without_litellm_returns_unavailable(self):
        def raise_import_error(*args, **kwargs):
            raise ImportError("No module named 'litellm'")

        with (
            patch.dict("sys.modules", {"litellm": None}),
            patch("builtins.__import__", side_effect=raise_import_error),
        ):
            analyzer = FailureAnalyzer()
            trace = self._make_trace()
            result = analyzer.analyze(trace, "Test expectations", ["test failure"])
            assert "unavailable" in result.lower()

    def test_analyze_run_with_failures(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Root cause: missing tool"))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            analyzer = FailureAnalyzer()
            trace = self._make_trace()
            scene = self._make_scene()

            run_data = {
                "trace": trace,
                "scene": scene,
                "check": {
                    "passed": False,
                    "checks": [
                        {
                            "label": "required_tool",
                            "passed": False,
                            "detail": "lookup_order NOT called",
                        }
                    ],
                },
                "metadata": {"run_id": "run_123", "scene_id": "test_scene"},
            }

            result = analyzer.analyze_run(run_data)

            assert isinstance(result, FailureAnalysis)
            assert result.run_id == "run_123"
            assert result.scene_id == "test_scene"
            assert "required_tool" in result.failed_checks
            assert result.analysis is not None

    def test_analyze_run_no_trace(self):
        analyzer = FailureAnalyzer()
        run_data = {
            "trace": None,
            "scene": None,
            "check": {
                "passed": False,
                "checks": [{"label": "test", "passed": False}],
            },
            "metadata": {"run_id": "run_123", "scene_id": "test_scene"},
        }

        result = analyzer.analyze_run(run_data)

        assert result.analysis == "No trace available for analysis."

    def test_analyze_run_no_failures(self):
        analyzer = FailureAnalyzer()
        trace = self._make_trace()

        run_data = {
            "trace": trace,
            "scene": None,
            "check": {
                "passed": True,
                "checks": [{"label": "test", "passed": True}],
            },
            "metadata": {"run_id": "run_123", "scene_id": "test_scene"},
        }

        result = analyzer.analyze_run(run_data)
        assert result.analysis == "No failures."

    def test_analyze_handles_none_content(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            analyzer = FailureAnalyzer()
            trace = self._make_trace()
            result = analyzer.analyze(trace, "Test expectations", ["test failure"])

            assert result == "No analysis available"


class TestPromptTemplates:
    def test_failure_analysis_prompt_template(self):
        assert "{expectations}" in FAILURE_ANALYSIS_PROMPT
        assert "{failed_checks}" in FAILURE_ANALYSIS_PROMPT

    def test_judge_system_prompt_template(self):
        assert "{rubric}" in JUDGE_SYSTEM_PROMPT
        assert "YES or NO" in JUDGE_SYSTEM_PROMPT
