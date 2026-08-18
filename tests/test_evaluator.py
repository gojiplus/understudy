"""Tests for the evaluator agent module."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from understudy.check import CheckItem, CheckResult
from understudy.evaluator.analyzer import AnalysisResult, DeepAnalyzer
from understudy.evaluator.coverage import (
    CapabilityCoverage,
    ConstraintCoverage,
    CoverageReport,
    CoverageTracker,
)
from understudy.evaluator.generator import SceneGenerator
from understudy.evaluator.inputs import (
    AgentSpec,
    Capability,
    CapabilityType,
    Constraint,
    ConstraintType,
    InputProcessor,
)
from understudy.evaluator.iteration import IterationController, IterationDecision, IterationState
from understudy.evaluator.report import EvaluationReport, EvaluationSummary, FailureReport
from understudy.models import Persona, PersonaPreset, Scene
from understudy.trace import ToolCall, Trace, Turn


class TestAgentSpec:
    """Tests for AgentSpec data model."""

    def test_create_agent_spec(self):
        spec = AgentSpec(
            name="test_agent",
            description="A test agent",
            capabilities=[
                Capability(
                    id="cap1",
                    name="Lookup Order",
                    description="Look up order by ID",
                    capability_type=CapabilityType.INFORMATION_RETRIEVAL,
                    tools_involved=["lookup_order"],
                )
            ],
            constraints=[
                Constraint(
                    id="con1",
                    name="No PII",
                    description="Never reveal customer PII",
                    constraint_type=ConstraintType.MUST_NOT_DO,
                )
            ],
            available_tools=["lookup_order", "update_order"],
        )

        assert spec.name == "test_agent"
        assert len(spec.capabilities) == 1
        assert len(spec.constraints) == 1
        assert "lookup_order" in spec.available_tools

    def test_capability_by_id(self):
        cap = Capability(id="cap1", name="Test", description="Test cap")
        spec = AgentSpec(name="test", capabilities=[cap])

        found = spec.capability_by_id("cap1")
        assert found is not None
        assert found.name == "Test"

        not_found = spec.capability_by_id("cap999")
        assert not_found is None

    def test_capabilities_by_type(self):
        caps = [
            Capability(
                id="cap1",
                name="Tool 1",
                description="",
                capability_type=CapabilityType.TOOL_USE,
            ),
            Capability(
                id="cap2",
                name="Tool 2",
                description="",
                capability_type=CapabilityType.TOOL_USE,
            ),
            Capability(
                id="cap3",
                name="Info",
                description="",
                capability_type=CapabilityType.INFORMATION_RETRIEVAL,
            ),
        ]
        spec = AgentSpec(name="test", capabilities=caps)

        tool_caps = spec.capabilities_by_type(CapabilityType.TOOL_USE)
        assert len(tool_caps) == 2

        info_caps = spec.capabilities_by_type(CapabilityType.INFORMATION_RETRIEVAL)
        assert len(info_caps) == 1

    def test_capabilities_by_tool(self):
        caps = [
            Capability(
                id="cap1",
                name="Lookup",
                description="",
                tools_involved=["lookup_order", "get_customer"],
            ),
            Capability(
                id="cap2",
                name="Update",
                description="",
                tools_involved=["update_order"],
            ),
        ]
        spec = AgentSpec(name="test", capabilities=caps)

        lookup_caps = spec.capabilities_by_tool("lookup_order")
        assert len(lookup_caps) == 1
        assert lookup_caps[0].id == "cap1"


class TestCoverageTracker:
    """Tests for coverage tracking."""

    @pytest.fixture
    def sample_spec(self):
        return AgentSpec(
            name="test_agent",
            capabilities=[
                Capability(
                    id="cap1",
                    name="Lookup Order",
                    description="Look up order",
                    tools_involved=["lookup_order"],
                    edge_cases=["invalid order ID", "cancelled order"],
                ),
                Capability(
                    id="cap2",
                    name="Process Refund",
                    description="Process refund",
                    tools_involved=["process_refund"],
                ),
            ],
            constraints=[
                Constraint(
                    id="con1",
                    name="No PII",
                    description="Never reveal PII",
                    constraint_type=ConstraintType.MUST_NOT_DO,
                ),
            ],
        )

    def test_initial_coverage(self, sample_spec):
        tracker = CoverageTracker(sample_spec)
        report = tracker.get_report()

        assert report.capability_coverage_rate == 0.0
        assert report.constraint_coverage_rate == 0.0
        assert len(report.untested_capabilities) == 2

    def test_record_scene_updates_coverage(self, sample_spec):
        tracker = CoverageTracker(sample_spec)

        scene = Scene(
            id="test_scene",
            starting_prompt="Test",
            conversation_plan="Test plan",
            persona=Persona.from_preset(PersonaPreset.COOPERATIVE),
            context={"_capability_id": "cap1"},
        )

        trace = Trace(
            scene_id="test_scene",
            turns=[
                Turn(role="user", content="Hello"),
                Turn(
                    role="agent",
                    content="Hi",
                    tool_calls=[
                        ToolCall(tool_name="lookup_order", arguments={"id": "123"})
                    ],
                ),
            ],
        )

        tracker.record_scene(scene, trace, passed=True)

        report = tracker.get_report()
        assert report.capability_coverage_rate == 0.5
        assert report.total_scenes_run == 1
        assert report.scenes_passed == 1

    def test_untested_capabilities(self, sample_spec):
        tracker = CoverageTracker(sample_spec)
        untested = tracker.get_untested_capabilities()
        assert len(untested) == 2

    def test_suggest_next_tests(self, sample_spec):
        tracker = CoverageTracker(sample_spec)
        suggestions = tracker.suggest_next_tests()

        assert len(suggestions) > 0
        assert all("type" in s for s in suggestions)
        assert all("reason" in s for s in suggestions)


class TestIterationController:
    """Tests for iteration control logic."""

    @pytest.fixture
    def controller(self):
        return IterationController(
            max_iterations=3,
            min_coverage=0.8,
            max_failure_rate=0.3,
        )

    @pytest.fixture
    def empty_coverage(self):
        return CoverageReport(
            capabilities=[],
            constraints=[],
            total_scenes_run=0,
            scenes_passed=0,
        )

    def test_stop_at_max_iterations(self, controller, empty_coverage):
        state = IterationState(
            iteration_number=3,
            max_iterations=3,
            scenes_run=10,
            scenes_passed=8,
            scenes_failed=2,
            coverage_rate=0.9,
            patterns_found=0,
            budget_remaining=0.0,
        )

        plan = controller.decide_next(state, empty_coverage, [])
        assert plan.decision == IterationDecision.STOP

    def test_drill_down_on_high_failure_rate(self, controller):
        coverage = CoverageReport(
            capabilities=[
                CapabilityCoverage(
                    capability_id="cap1",
                    capability_name="Test",
                    tested=True,
                    passed=False,
                )
            ],
            constraints=[],
            total_scenes_run=10,
            scenes_passed=5,
        )

        state = IterationState(
            iteration_number=1,
            max_iterations=3,
            scenes_run=10,
            scenes_passed=5,
            scenes_failed=5,
            coverage_rate=0.5,
            patterns_found=0,
            budget_remaining=0.67,
        )

        results = [
            AnalysisResult(
                scene_id="scene1",
                passed=False,
                failure_type="check_failed",
                failure_description="Test failed",
                affected_capability="cap1",
            )
        ]

        plan = controller.decide_next(state, coverage, results)
        assert plan.decision == IterationDecision.DRILL_DOWN

    def test_expand_coverage_when_low(self, controller):
        coverage = CoverageReport(
            capabilities=[
                CapabilityCoverage(
                    capability_id="cap1",
                    capability_name="Tested",
                    tested=True,
                    passed=True,
                ),
                CapabilityCoverage(
                    capability_id="cap2",
                    capability_name="Untested",
                    tested=False,
                    passed=False,
                ),
            ],
            constraints=[],
            total_scenes_run=5,
            scenes_passed=5,
        )

        state = IterationState(
            iteration_number=1,
            max_iterations=3,
            scenes_run=5,
            scenes_passed=5,
            scenes_failed=0,
            coverage_rate=0.5,
            patterns_found=0,
            budget_remaining=0.67,
        )

        plan = controller.decide_next(state, coverage, [])
        assert plan.decision == IterationDecision.EXPAND_COVERAGE

    def test_adversarial_probe_when_high_coverage_moderate_failures(self, controller):
        coverage = CoverageReport(
            capabilities=[
                CapabilityCoverage(
                    capability_id="cap1",
                    capability_name="Test",
                    tested=True,
                    passed=True,
                )
            ],
            constraints=[],
            total_scenes_run=10,
            scenes_passed=8,
        )

        state = IterationState(
            iteration_number=1,
            max_iterations=3,
            scenes_run=10,
            scenes_passed=8,
            scenes_failed=2,
            coverage_rate=1.0,
            patterns_found=0,
            budget_remaining=0.67,
        )

        plan = controller.decide_next(state, coverage, [])
        assert plan.decision == IterationDecision.ADVERSARIAL_PROBE

    def test_stop_when_coverage_achieved_and_low_failures(self, controller):
        coverage = CoverageReport(
            capabilities=[
                CapabilityCoverage(
                    capability_id="cap1",
                    capability_name="Test",
                    tested=True,
                    passed=True,
                )
            ],
            constraints=[],
            total_scenes_run=10,
            scenes_passed=10,
        )

        state = IterationState(
            iteration_number=1,
            max_iterations=3,
            scenes_run=10,
            scenes_passed=10,
            scenes_failed=0,
            coverage_rate=1.0,
            patterns_found=0,
            budget_remaining=0.67,
        )

        plan = controller.decide_next(state, coverage, [])
        assert plan.decision == IterationDecision.STOP


class TestEvaluationReport:
    """Tests for evaluation report generation."""

    def test_report_passed_when_no_failures(self):
        report = EvaluationReport(
            agent_name="test_agent",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            failures=[],
        )
        assert report.passed is True

    def test_report_failed_when_has_failures(self):
        report = EvaluationReport(
            agent_name="test_agent",
            started_at=datetime.now(UTC),
            failures=[
                FailureReport(
                    scene_id="scene1",
                    capability_id="cap1",
                    failure_type="check_failed",
                    description="Test failed",
                )
            ],
        )
        assert report.passed is False

    def test_report_to_text(self):
        report = EvaluationReport(
            agent_name="test_agent",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            summary=EvaluationSummary(
                total_scenes=10,
                scenes_passed=8,
                scenes_failed=2,
            ),
            failures=[
                FailureReport(
                    scene_id="scene1",
                    capability_id=None,
                    failure_type="check_failed",
                    description="Required tool not called",
                    root_cause="Agent skipped verification step",
                )
            ],
        )

        text = report.to_text()
        assert "test_agent" in text
        assert "80.0%" in text
        assert "Required tool not called" in text

    def test_report_to_dict(self):
        started = datetime.now(UTC)
        report = EvaluationReport(
            agent_name="test_agent",
            started_at=started,
            summary=EvaluationSummary(total_scenes=5),
        )

        data = report.to_dict()
        assert data["agent_name"] == "test_agent"
        assert "summary" in data
        assert data["summary"]["total_scenes"] == 5


class TestSceneGenerator:
    """Tests for scene generation (with mocked LLM)."""

    @pytest.fixture
    def sample_spec(self):
        return AgentSpec(
            name="test_agent",
            description="A test agent",
            capabilities=[
                Capability(
                    id="cap1",
                    name="Lookup Order",
                    description="Look up order by ID",
                    tools_involved=["lookup_order"],
                    edge_cases=["invalid ID"],
                )
            ],
            constraints=[
                Constraint(
                    id="con1",
                    name="No PII",
                    description="Never reveal PII",
                    constraint_type=ConstraintType.MUST_NOT_DO,
                )
            ],
            available_tools=["lookup_order"],
        )

    @patch("litellm.completion")
    def test_generate_for_capability(self, mock_completion, sample_spec):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """{
            "id": "test_lookup_happy",
            "description": "Test basic order lookup",
            "starting_prompt": "Can you look up order ORD-123?",
            "conversation_plan": "User asks for order, agent looks it up",
            "persona": "cooperative",
            "context": {},
            "expectations": {
                "required_tools": ["lookup_order"],
                "expected_resolution": "completed"
            }
        }"""
        mock_completion.return_value = mock_response

        generator = SceneGenerator(model="test-model")
        cap = sample_spec.capabilities[0]

        scenes = generator.generate_for_capability(
            sample_spec, cap, include_edge_cases=False
        )

        assert len(scenes) >= 1
        assert scenes[0].id == "test_lookup_happy"
        assert "lookup_order" in scenes[0].expectations.required_tools

    @patch("litellm.completion")
    def test_generate_for_constraint(self, mock_completion, sample_spec):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """{
            "id": "test_pii_compliance",
            "description": "Test PII compliance",
            "starting_prompt": "Can I get my order status?",
            "conversation_plan": "User asks for order without trying to get PII",
            "persona": "cooperative",
            "context": {},
            "expectations": {
                "required_tools": ["lookup_order"],
                "expected_resolution": "completed"
            }
        }"""
        mock_completion.return_value = mock_response

        generator = SceneGenerator(model="test-model")
        con = sample_spec.constraints[0]

        scenes = generator.generate_for_constraint(sample_spec, con)

        assert len(scenes) == 2


class TestDeepAnalyzer:
    """Tests for deep analysis (with mocked LLM)."""

    @patch("litellm.completion")
    def test_analyze_passing_trace(self, mock_completion):
        analyzer = DeepAnalyzer(model="test-model")

        trace = Trace(
            scene_id="test_scene",
            turns=[Turn(role="user", content="Hello")],
        )
        check_result = CheckResult(checks=[])

        result = analyzer.analyze_trace(trace, check_result)

        assert result.passed is True
        assert result.scene_id == "test_scene"
        mock_completion.assert_not_called()

    @patch("litellm.completion")
    def test_analyze_failing_trace(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """{
            "failure_type": "check_failed",
            "failure_description": "Required tool was not called",
            "root_cause": "Agent did not verify the order",
            "suggested_fix": "Add verification step to agent logic",
            "severity": "error",
            "trace_highlights": ["Turn 2: Agent responded without checking"]
        }"""
        mock_completion.return_value = mock_response

        analyzer = DeepAnalyzer(model="test-model")

        trace = Trace(
            scene_id="test_scene",
            turns=[
                Turn(role="user", content="Check my order"),
                Turn(role="agent", content="Your order is fine"),
            ],
        )
        check_result = CheckResult(
            checks=[
                CheckItem(
                    label="required_tool",
                    passed=False,
                    detail="lookup_order NOT called",
                )
            ]
        )

        result = analyzer.analyze_trace(trace, check_result)

        assert result.passed is False
        assert result.failure_type == "check_failed"
        assert "Required tool" in result.failure_description
        assert result.root_cause is not None


class TestInputProcessor:
    """Tests for input processing (with mocked LLM)."""

    @patch("litellm.completion")
    def test_process_spec_only(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = """{
            "name": "CustomerServiceAgent",
            "description": "Handles customer inquiries",
            "capabilities": [
                {
                    "id": "lookup",
                    "name": "Order Lookup",
                    "description": "Look up customer orders",
                    "type": "information_retrieval",
                    "tools_involved": ["lookup_order"]
                }
            ],
            "constraints": [
                {
                    "id": "pii",
                    "name": "PII Protection",
                    "description": "Never expose customer PII",
                    "type": "must_not_do"
                }
            ],
            "available_tools": ["lookup_order", "update_order"]
        }"""
        mock_completion.return_value = mock_response

        processor = InputProcessor(model="test-model")
        spec = processor.process(spec="Customer service agent that handles orders")

        assert spec.name == "CustomerServiceAgent"
        assert len(spec.capabilities) == 1
        assert len(spec.constraints) == 1
        assert "lookup_order" in spec.available_tools

    def test_process_requires_input(self):
        processor = InputProcessor(model="test-model")

        with pytest.raises(ValueError, match="At least one input source"):
            processor.process()


class TestCoverageReport:
    """Tests for coverage report."""

    def test_coverage_rates(self):
        report = CoverageReport(
            capabilities=[
                CapabilityCoverage("cap1", "Test1", tested=True, passed=True),
                CapabilityCoverage("cap2", "Test2", tested=True, passed=False),
                CapabilityCoverage("cap3", "Test3", tested=False, passed=False),
            ],
            constraints=[
                ConstraintCoverage("con1", "Con1", compliance_tested=True),
                ConstraintCoverage("con2", "Con2", compliance_tested=False),
            ],
            total_scenes_run=10,
            scenes_passed=7,
        )

        assert report.capability_coverage_rate == pytest.approx(2 / 3)
        assert report.constraint_coverage_rate == 0.5
        assert len(report.untested_capabilities) == 1
        assert len(report.failing_capabilities) == 1

    def test_coverage_to_text(self):
        report = CoverageReport(
            capabilities=[
                CapabilityCoverage("cap1", "Test1", tested=True, passed=True),
            ],
            constraints=[],
            total_scenes_run=5,
            scenes_passed=5,
        )

        text = report.to_text()
        assert "Coverage Report" in text
        assert "Test1" in text
        assert "PASS" in text
