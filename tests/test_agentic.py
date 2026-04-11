"""Unit tests for the agentic flow evaluation module."""

import json

import yaml

from understudy.agentic import (
    AgenticCheckItem,
    AgenticCheckResult,
    AgenticExpectations,
    AgenticScene,
    AgenticTrace,
    Artifact,
    Step,
    StepResult,
    Task,
    check_agentic,
    run_agentic,
)
from understudy.agentic.metrics import (
    action_efficiency,
    action_trajectory,
    compute_all_metrics,
    goal_completion,
    reasoning_quality,
)
from understudy.mocks import MockToolkit


class TestTask:
    def test_basic_task(self):
        task = Task(
            description="Write a function",
            goal="Function passes all tests",
        )
        assert task.description == "Write a function"
        assert task.goal == "Function passes all tests"
        assert task.max_steps == 100
        assert task.constraints == []

    def test_task_with_constraints(self):
        task = Task(
            description="Refactor code",
            goal="Code refactored",
            constraints=["Don't change behavior", "Keep tests passing"],
            initial_state={"file": "main.py"},
            max_steps=50,
            max_tokens=10000,
        )
        assert len(task.constraints) == 2
        assert task.initial_state["file"] == "main.py"
        assert task.max_steps == 50
        assert task.max_tokens == 10000


class TestStep:
    def test_think_step(self):
        step = Step(
            step_number=1,
            step_type="think",
            reasoning="I need to read the file first",
        )
        assert step.step_type == "think"
        assert step.reasoning == "I need to read the file first"
        assert step.action is None

    def test_act_step(self):
        step = Step(
            step_number=2,
            step_type="act",
            action="read_file",
            action_args={"path": "/tmp/test.py"},
            observation="def foo(): pass",
            tokens_used=100,
            latency_ms=500,
        )
        assert step.step_type == "act"
        assert step.action == "read_file"
        assert step.action_args["path"] == "/tmp/test.py"
        assert step.observation == "def foo(): pass"

    def test_observe_step(self):
        step = Step(
            step_number=3,
            step_type="observe",
            observation="Test passed",
        )
        assert step.step_type == "observe"
        assert step.observation == "Test passed"


class TestAgenticTrace:
    def _make_trace(self) -> AgenticTrace:
        task = Task(description="Write tests", goal="All tests pass")
        return AgenticTrace(
            scene_id="test_scene",
            task=task,
            steps=[
                Step(step_number=1, step_type="think", reasoning="Plan the approach"),
                Step(
                    step_number=2,
                    step_type="act",
                    action="read_file",
                    action_args={"path": "main.py"},
                    observation="def foo(): pass",
                    tokens_used=100,
                ),
                Step(step_number=3, step_type="think", reasoning="Now write tests"),
                Step(
                    step_number=4,
                    step_type="act",
                    action="write_file",
                    action_args={"path": "test_main.py", "content": "def test_foo(): ..."},
                    observation="File written",
                    tokens_used=150,
                ),
                Step(
                    step_number=5,
                    step_type="act",
                    action="run_tests",
                    action_args={},
                    observation="All tests passed",
                    tokens_used=50,
                ),
            ],
            outcome="success",
        )

    def test_total_steps(self):
        trace = self._make_trace()
        assert trace.total_steps == 5

    def test_total_tokens(self):
        trace = self._make_trace()
        assert trace.total_tokens == 300

    def test_action_sequence(self):
        trace = self._make_trace()
        assert trace.action_sequence() == ["read_file", "write_file", "run_tests"]

    def test_performed(self):
        trace = self._make_trace()
        assert trace.performed("read_file")
        assert trace.performed("write_file")
        assert not trace.performed("delete_file")

    def test_performed_with_args(self):
        trace = self._make_trace()
        assert trace.performed("read_file", path="main.py")
        assert not trace.performed("read_file", path="other.py")

    def test_actions_to(self):
        trace = self._make_trace()
        actions = trace.actions_to("read_file")
        assert len(actions) == 1
        assert actions[0].action_args["path"] == "main.py"

    def test_thinking_steps(self):
        trace = self._make_trace()
        thinking = trace.thinking_steps()
        assert len(thinking) == 2
        assert all(s.step_type == "think" for s in thinking)

    def test_tool_calls_compatibility(self):
        trace = self._make_trace()
        calls = trace.tool_calls
        assert len(calls) == 3
        assert calls[0].tool_name == "read_file"
        assert calls[1].tool_name == "write_file"

    def test_conversation_text(self):
        trace = self._make_trace()
        text = trace.conversation_text()
        assert "TASK:" in text
        assert "GOAL:" in text
        assert "[THINK]:" in text
        assert "[ACT]:" in text
        assert "OUTCOME:" in text

    def test_retry_count(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="retry_test",
            task=task,
            steps=[
                Step(
                    step_number=1,
                    step_type="act",
                    action="api_call",
                    error="Connection failed",
                ),
                Step(
                    step_number=2,
                    step_type="act",
                    action="api_call",
                    error="Timeout",
                ),
                Step(
                    step_number=3,
                    step_type="act",
                    action="api_call",
                    observation="Success",
                ),
            ],
            outcome="success",
        )
        assert trace.retry_count() == 2


class TestAgenticTraceFromJson:
    def test_from_json(self):
        data = {
            "scene_id": "json_scene",
            "task": {
                "description": "Load from JSON",
                "goal": "Trace loaded",
            },
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "act",
                    "action": "test_action",
                    "action_args": {"key": "value"},
                    "observation": "result",
                    "tokens_used": 50,
                }
            ],
            "outcome": "success",
            "final_state": {"completed": True},
        }
        trace = AgenticTrace.from_json(data)
        assert trace.scene_id == "json_scene"
        assert trace.task.description == "Load from JSON"
        assert len(trace.steps) == 1
        assert trace.outcome == "success"

    def test_from_file(self, tmp_path):
        data = {
            "scene_id": "file_scene",
            "task": {
                "description": "Load from file",
                "goal": "Trace loaded from file",
            },
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "think",
                    "reasoning": "Planning",
                }
            ],
            "outcome": "completed",
        }
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(data))

        trace = AgenticTrace.from_file(path)
        assert trace.scene_id == "file_scene"
        assert trace.task.goal == "Trace loaded from file"


class TestAgenticScene:
    def test_from_yaml(self, tmp_path):
        scene_data = {
            "id": "test_scene",
            "description": "A test scene",
            "task": {
                "description": "Write tests",
                "goal": "Tests pass",
                "constraints": ["No side effects"],
                "max_steps": 30,
            },
            "environment": {"working_dir": "/tmp"},
            "expectations": {
                "required_actions": ["read_file", "write_file"],
                "forbidden_actions": ["delete_file"],
                "max_steps": 25,
            },
        }
        path = tmp_path / "scene.yaml"
        path.write_text(yaml.dump(scene_data))

        scene = AgenticScene.from_file(path)
        assert scene.id == "test_scene"
        assert scene.task.description == "Write tests"
        assert scene.task.max_steps == 30
        assert "read_file" in scene.expectations.required_actions
        assert "delete_file" in scene.expectations.forbidden_actions

    def test_from_json(self, tmp_path):
        scene_data = {
            "id": "json_scene",
            "task": {
                "description": "JSON task",
                "goal": "Load JSON scene",
            },
        }
        path = tmp_path / "scene.json"
        path.write_text(json.dumps(scene_data))

        scene = AgenticScene.from_file(path)
        assert scene.id == "json_scene"


class TestAgenticExpectations:
    def test_defaults(self):
        exp = AgenticExpectations()
        assert exp.goal_predicate is None
        assert exp.max_steps is None
        assert exp.required_actions == []
        assert exp.forbidden_actions == []

    def test_with_values(self):
        exp = AgenticExpectations(
            goal_predicate="outcome == 'success'",
            max_steps=50,
            max_tokens=5000,
            max_retries=3,
            required_actions=["read_file", "write_file"],
            forbidden_actions=["delete:*"],
        )
        assert exp.goal_predicate == "outcome == 'success'"
        assert exp.max_steps == 50
        assert len(exp.required_actions) == 2


class TestCheckAgentic:
    def _make_trace(self, outcome="success", steps=None) -> AgenticTrace:
        task = Task(description="Test", goal="Goal")
        if steps is None:
            steps = [
                Step(step_number=1, step_type="act", action="read_file", tokens_used=50),
                Step(step_number=2, step_type="act", action="write_file", tokens_used=50),
            ]
        return AgenticTrace(
            scene_id="check_test",
            task=task,
            steps=steps,
            outcome=outcome,
        )

    def test_all_pass(self):
        trace = self._make_trace()
        exp = AgenticExpectations(
            required_actions=["read_file", "write_file"],
            max_steps=10,
        )
        result = check_agentic(trace, exp)
        assert result.passed

    def test_goal_predicate_pass(self):
        trace = self._make_trace(outcome="success")
        exp = AgenticExpectations(goal_predicate="outcome == 'success'")
        result = check_agentic(trace, exp)
        assert result.passed

    def test_goal_predicate_fail(self):
        trace = self._make_trace(outcome="failure")
        exp = AgenticExpectations(goal_predicate="outcome == 'success'")
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_required_actions_pass(self):
        trace = self._make_trace()
        exp = AgenticExpectations(required_actions=["read_file"])
        result = check_agentic(trace, exp)
        assert result.passed

    def test_required_actions_fail(self):
        trace = self._make_trace()
        exp = AgenticExpectations(required_actions=["delete_file"])
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_forbidden_actions_pass(self):
        trace = self._make_trace()
        exp = AgenticExpectations(forbidden_actions=["delete_file"])
        result = check_agentic(trace, exp)
        assert result.passed

    def test_forbidden_actions_fail(self):
        trace = self._make_trace()
        exp = AgenticExpectations(forbidden_actions=["read_file"])
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_forbidden_actions_wildcard(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="wildcard_test",
            task=task,
            steps=[
                Step(step_number=1, step_type="act", action="delete_file"),
                Step(step_number=2, step_type="act", action="delete_dir"),
            ],
            outcome="success",
        )
        exp = AgenticExpectations(forbidden_actions=["delete*"])
        result = check_agentic(trace, exp)
        assert not result.passed
        violations = [c for c in result.failed_checks if c.label == "forbidden_action"]
        assert len(violations) == 1

    def test_max_steps_pass(self):
        trace = self._make_trace()
        exp = AgenticExpectations(max_steps=10)
        result = check_agentic(trace, exp)
        assert result.passed

    def test_max_steps_fail(self):
        trace = self._make_trace()
        exp = AgenticExpectations(max_steps=1)
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_max_tokens_pass(self):
        trace = self._make_trace()
        exp = AgenticExpectations(max_tokens=200)
        result = check_agentic(trace, exp)
        assert result.passed

    def test_max_tokens_fail(self):
        trace = self._make_trace()
        exp = AgenticExpectations(max_tokens=50)
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_max_retries_pass(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="retry_test",
            task=task,
            steps=[
                Step(step_number=1, step_type="act", action="api", error="Fail"),
                Step(step_number=2, step_type="act", action="api", observation="OK"),
            ],
            outcome="success",
        )
        exp = AgenticExpectations(max_retries=2)
        result = check_agentic(trace, exp)
        assert result.passed

    def test_max_retries_fail(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="retry_test",
            task=task,
            steps=[
                Step(step_number=1, step_type="act", action="api", error="Fail"),
                Step(step_number=2, step_type="act", action="api", error="Fail"),
                Step(step_number=3, step_type="act", action="api", observation="OK"),
            ],
            outcome="success",
        )
        exp = AgenticExpectations(max_retries=1)
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_golden_output_pass(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="golden_test",
            task=task,
            steps=[],
            outcome="success",
            final_state={"result": 42, "status": "done"},
        )
        exp = AgenticExpectations(golden_output={"result": 42})
        result = check_agentic(trace, exp)
        assert result.passed

    def test_golden_output_fail(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="golden_test",
            task=task,
            steps=[],
            outcome="success",
            final_state={"result": 0},
        )
        exp = AgenticExpectations(golden_output={"result": 42})
        result = check_agentic(trace, exp)
        assert not result.passed

    def test_summary(self):
        trace = self._make_trace()
        exp = AgenticExpectations(
            required_actions=["read_file"],
            forbidden_actions=["delete_file"],
        )
        result = check_agentic(trace, exp)
        summary = result.summary()
        assert "+" in summary
        assert "required_action" in summary


class TestAgenticMetrics:
    def _make_trace(self, outcome="success") -> AgenticTrace:
        task = Task(description="Test", goal="Goal")
        return AgenticTrace(
            scene_id="metrics_test",
            task=task,
            steps=[
                Step(step_number=1, step_type="think", reasoning="Planning approach"),
                Step(
                    step_number=2,
                    step_type="act",
                    action="read",
                    tokens_used=100,
                    latency_ms=500,
                ),
                Step(step_number=3, step_type="think", reasoning="Analyze results"),
                Step(
                    step_number=4,
                    step_type="act",
                    action="write",
                    tokens_used=150,
                    latency_ms=600,
                ),
            ],
            outcome=outcome,
        )

    def test_goal_completion_success(self):
        trace = self._make_trace(outcome="success")
        result = goal_completion(trace)
        assert result.name == "goal_completion"
        assert result.value == 1.0
        assert result.passed is True

    def test_goal_completion_failure(self):
        trace = self._make_trace(outcome="failed")
        result = goal_completion(trace)
        assert result.value == 0.0
        assert result.passed is False

    def test_reasoning_quality(self):
        trace = self._make_trace()
        result = reasoning_quality(trace)
        assert result.name == "reasoning_quality"
        assert result.value["thinking_steps"] == 2
        assert result.value["action_steps"] == 2
        assert result.value["think_act_ratio"] == 1.0

    def test_reasoning_quality_no_thinking(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="no_think",
            task=task,
            steps=[
                Step(step_number=1, step_type="act", action="do"),
            ],
            outcome="success",
        )
        result = reasoning_quality(trace)
        assert result.value["thinking_steps"] == 0
        assert result.value["think_act_ratio"] == 0.0

    def test_action_efficiency(self):
        trace = self._make_trace()
        result = action_efficiency(trace)
        assert result.name == "action_efficiency"
        assert result.value["total_steps"] == 4
        assert result.value["action_steps"] == 2
        assert result.value["total_tokens"] == 250
        assert result.value["total_latency_ms"] == 1100

    def test_action_efficiency_with_expectations(self):
        trace = self._make_trace()
        exp = AgenticExpectations(max_steps=10, max_tokens=500)
        result = action_efficiency(trace, exp)
        assert result.passed is True

    def test_action_trajectory(self):
        trace = self._make_trace()
        result = action_trajectory(trace)
        assert result.name == "action_trajectory"
        assert result.value["sequence"] == ["read", "write"]
        assert result.value["total_actions"] == 2
        assert set(result.value["unique_actions"]) == {"read", "write"}

    def test_compute_all_metrics(self):
        trace = self._make_trace()
        metrics = compute_all_metrics(trace)
        assert "goal_completion" in metrics
        assert "reasoning_quality" in metrics
        assert "action_efficiency" in metrics
        assert "action_trajectory" in metrics


class MockAgenticApp:
    """Mock implementation of AgenticApp for testing."""

    def __init__(
        self,
        steps: list[StepResult] | None = None,
        max_steps: int = 3,
    ):
        self.steps_to_return = steps or [
            StepResult(step_type="think", reasoning="Planning"),
            StepResult(
                step_type="act",
                action="test_action",
                action_args={"key": "value"},
                observation="Result",
                tokens_used=50,
            ),
        ]
        self.max_steps = max_steps
        self.step_count = 0
        self._done = False

    def start(self, task: Task, environment: dict | None = None):
        self.task = task
        self.environment = environment
        self.step_count = 0
        self._done = False

    def step(self) -> StepResult:
        if self.step_count < len(self.steps_to_return):
            result = self.steps_to_return[self.step_count]
            self.step_count += 1
            if self.step_count >= len(self.steps_to_return):
                self._done = True
            return result
        self._done = True
        return StepResult(step_type="observe", observation="Done")

    def is_done(self) -> bool:
        return self._done or self.step_count >= self.max_steps

    def get_outcome(self) -> str:
        return "success"

    def get_state(self) -> dict:
        return {"completed": True}

    def stop(self):
        pass


class TestRunAgentic:
    def test_basic_run(self):
        app = MockAgenticApp()
        scene = AgenticScene(
            id="run_test",
            task=Task(description="Test task", goal="Test goal"),
        )
        trace = run_agentic(app, scene)

        assert trace.scene_id == "run_test"
        assert trace.outcome == "success"
        assert trace.total_steps == 2

    def test_run_with_mocks(self):
        mocks = MockToolkit()

        @mocks.handle("mocked_action")
        def mocked_action(key: str) -> str:
            return f"Mocked result for {key}"

        app = MockAgenticApp(
            steps=[
                StepResult(
                    step_type="act",
                    action="mocked_action",
                    action_args={"key": "test"},
                )
            ]
        )
        scene = AgenticScene(
            id="mock_test",
            task=Task(description="Test", goal="Goal"),
        )
        trace = run_agentic(app, scene, mocks=mocks)

        assert trace.steps[0].observation == "Mocked result for test"

    def test_max_steps_exceeded(self):
        app = MockAgenticApp(
            steps=[
                StepResult(step_type="act", action="action1"),
                StepResult(step_type="act", action="action2"),
                StepResult(step_type="act", action="action3"),
                StepResult(step_type="act", action="action4"),
            ],
            max_steps=100,
        )
        scene = AgenticScene(
            id="max_steps_test",
            task=Task(description="Test", goal="Goal", max_steps=2),
        )
        trace = run_agentic(app, scene)

        assert trace.outcome == "max_steps_exceeded"
        assert trace.total_steps == 2

    def test_run_records_timing(self):
        app = MockAgenticApp()
        scene = AgenticScene(
            id="timing_test",
            task=Task(description="Test", goal="Goal"),
        )
        trace = run_agentic(app, scene)

        assert trace.started_at is not None
        assert trace.finished_at is not None
        for step in trace.steps:
            assert step.latency_ms >= 0


class TestAgenticCheckResult:
    def test_passed_all(self):
        result = AgenticCheckResult(
            checks=[
                AgenticCheckItem(label="check1", passed=True, detail="OK"),
                AgenticCheckItem(label="check2", passed=True, detail="OK"),
            ]
        )
        assert result.passed

    def test_passed_some_failed(self):
        result = AgenticCheckResult(
            checks=[
                AgenticCheckItem(label="check1", passed=True, detail="OK"),
                AgenticCheckItem(label="check2", passed=False, detail="FAIL"),
            ]
        )
        assert not result.passed

    def test_failed_checks(self):
        result = AgenticCheckResult(
            checks=[
                AgenticCheckItem(label="check1", passed=True, detail="OK"),
                AgenticCheckItem(label="check2", passed=False, detail="FAIL"),
            ]
        )
        failed = result.failed_checks
        assert len(failed) == 1
        assert failed[0].label == "check2"


class TestArtifact:
    def test_basic_artifact(self):
        artifact = Artifact(
            name="output.txt",
            artifact_type="file",
            content="Hello, world!",
            path="/tmp/output.txt",
        )
        assert artifact.name == "output.txt"
        assert artifact.artifact_type == "file"
        assert artifact.content == "Hello, world!"

    def test_artifact_in_trace(self):
        task = Task(description="Test", goal="Goal")
        trace = AgenticTrace(
            scene_id="artifact_test",
            task=task,
            steps=[],
            outcome="success",
            artifacts=[
                Artifact(name="result.json", artifact_type="json", content={"key": "value"}),
            ],
        )
        assert len(trace.artifacts) == 1
        assert trace.artifacts[0].name == "result.json"
