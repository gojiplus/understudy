"""Evaluator Agent: LLM-powered evaluation of AI agents.

The main EvaluatorAgent class that orchestrates the full evaluation loop.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..mocks import MockToolkit
from ..models import Scene
from ..runner import AgentApp
from ..trace import Trace
from .analyzer import AnalysisResult, DeepAnalyzer, PatternAnalysis
from .generator import GenerationMode, SceneGenerator
from .inputs import AgentSpec, InputProcessor
from .iteration import IterationController, IterationDecision, IterationPlan
from .report import EvaluationReport, EvaluationSummary, FailureReport
from .tools import EvaluatorToolkit


class EvaluatorAgent:
    """An agent that evaluates other agents.

    Uses LLM intelligence to:
    - Understand agents from code, docs, specs, and traces
    - Generate comprehensive test scenarios
    - Execute tests and analyze results deeply
    - Iterate: drill down on failures, expand coverage
    - Report what's broken and why

    Example:
        ```python
        from understudy.evaluator import EvaluatorAgent

        evaluator = EvaluatorAgent(
            target_app=my_agent,
            model="claude-sonnet-4-20250514",
        )

        report = evaluator.evaluate(
            spec="This agent handles customer returns. It should...",
            max_iterations=3,
            adversarial=True,
        )

        print(report.summary)
        print(report.failures)
        ```
    """

    def __init__(
        self,
        target_app: AgentApp,
        model: str = "claude-sonnet-4-20250514",
        simulator_model: str = "gpt-4o",
        mocks: MockToolkit | None = None,
    ):
        """Initialize the evaluator agent.

        Args:
            target_app: The agent application to evaluate
            model: Model for LLM operations (generation, analysis)
            simulator_model: Model for the user simulator
            mocks: Optional mock toolkit for tool responses
        """
        self.target_app = target_app
        self.model = model
        self.simulator_model = simulator_model
        self.mocks = mocks

        self.input_processor = InputProcessor(model=model)
        self.generator = SceneGenerator(model=model)
        self.analyzer = DeepAnalyzer(model=model)

    def evaluate(
        self,
        code: str | Path | None = None,
        docs: str | Path | None = None,
        spec: str | None = None,
        golden_traces: list[Trace] | None = None,
        existing_scenes: list[Scene] | None = None,
        max_iterations: int = 3,
        adversarial: bool = True,
        min_coverage: float = 0.8,
        output_dir: Path | str | None = None,
    ) -> EvaluationReport:
        """Run a full evaluation of the target agent.

        Args:
            code: Path to agent code directory or code string
            docs: Path to documentation directory or docs string
            spec: Natural language specification
            golden_traces: List of known-good execution traces
            existing_scenes: Existing Scene objects to use/learn from
            max_iterations: Maximum evaluation iterations
            adversarial: Whether to include adversarial testing
            min_coverage: Minimum coverage rate target
            output_dir: Optional directory to save generated scenes

        Returns:
            Complete evaluation report
        """
        started_at = datetime.now(UTC)

        agent_spec = self.input_processor.process(
            code=code,
            docs=docs,
            spec=spec,
            golden_traces=golden_traces,
            existing_scenes=existing_scenes,
        )

        toolkit = EvaluatorToolkit(
            target_app=self.target_app,
            spec=agent_spec,
            mocks=self.mocks,
            model=self.model,
            simulator_model=self.simulator_model,
        )

        controller = IterationController(
            max_iterations=max_iterations,
            min_coverage=min_coverage,
        )

        all_scenes: list[Scene] = list(existing_scenes) if existing_scenes else []
        analysis_results: list[AnalysisResult] = []
        iteration_log: list[dict[str, Any]] = []

        for iteration in range(max_iterations):
            coverage = toolkit.coverage_tracker.get_report()
            state = controller.get_iteration_state(
                iteration_number=iteration,
                scenes_run=coverage.total_scenes_run,
                scenes_passed=coverage.scenes_passed,
                coverage=coverage,
                patterns_found=0,
            )

            plan = controller.decide_next(
                state=state,
                coverage=coverage,
                analysis_results=analysis_results,
            )

            iteration_log.append(
                {
                    "iteration": iteration,
                    "decision": plan.decision.value,
                    "reason": plan.reason,
                    "coverage_rate": coverage.overall_coverage_rate,
                }
            )

            if plan.decision == IterationDecision.STOP:
                break

            scenes = self._generate_scenes_for_plan(
                toolkit, agent_spec, plan, adversarial and iteration > 0
            )
            all_scenes.extend(scenes)

            if output_dir:
                toolkit.save_scenes(scenes, Path(output_dir))

            for scene in scenes:
                result = toolkit.run_scene(scene)
                if result.data and not result.data.passed:
                    analysis = toolkit.analyze_trace(
                        trace=result.data.trace,
                        check_result=result.data.check_result,
                        scene_context=scene.context,
                    )
                    if analysis.data:
                        analysis_results.append(analysis.data)

        pattern_result = toolkit.analyze_patterns(analysis_results)
        pattern_analysis = pattern_result.data if pattern_result.success else None

        report = self._build_report(
            agent_spec=agent_spec,
            started_at=started_at,
            toolkit=toolkit,
            analysis_results=analysis_results,
            pattern_analysis=pattern_analysis,
            all_scenes=all_scenes,
            iteration_log=iteration_log,
        )

        return report

    def evaluate_scenes(
        self,
        scenes: list[Scene],
        spec: str | None = None,
        analyze_failures: bool = True,
    ) -> EvaluationReport:
        """Evaluate the agent using provided scenes only.

        Simpler than full evaluate() - just runs the given scenes
        without generation or iteration.

        Args:
            scenes: Scenes to run
            spec: Optional natural language spec for context
            analyze_failures: Whether to deeply analyze failures

        Returns:
            Evaluation report
        """
        started_at = datetime.now(UTC)

        agent_spec = (
            self.input_processor.process(spec=spec)
            if spec
            else AgentSpec(name="unknown")
        )

        toolkit = EvaluatorToolkit(
            target_app=self.target_app,
            spec=agent_spec,
            mocks=self.mocks,
            model=self.model,
            simulator_model=self.simulator_model,
        )

        analysis_results: list[AnalysisResult] = []

        for scene in scenes:
            result = toolkit.run_scene(scene)
            if analyze_failures and result.data and not result.data.passed:
                analysis = toolkit.analyze_trace(
                    trace=result.data.trace,
                    check_result=result.data.check_result,
                    scene_context=scene.context,
                )
                if analysis.data:
                    analysis_results.append(analysis.data)

        pattern_analysis = None
        if analyze_failures and analysis_results:
            pattern_result = toolkit.analyze_patterns(analysis_results)
            pattern_analysis = pattern_result.data if pattern_result.success else None

        return self._build_report(
            agent_spec=agent_spec,
            started_at=started_at,
            toolkit=toolkit,
            analysis_results=analysis_results,
            pattern_analysis=pattern_analysis,
            all_scenes=scenes,
            iteration_log=[],
        )

    def generate_scenes(
        self,
        code: str | Path | None = None,
        docs: str | Path | None = None,
        spec: str | None = None,
        mode: GenerationMode = GenerationMode.SYSTEMATIC,
        max_scenes: int = 10,
        output_dir: Path | str | None = None,
    ) -> list[Scene]:
        """Generate test scenes without running them.

        Useful for creating reusable test fixtures.

        Args:
            code: Path to agent code directory or code string
            docs: Path to documentation directory or docs string
            spec: Natural language specification
            mode: Generation mode
            max_scenes: Maximum scenes to generate
            output_dir: Optional directory to save scenes

        Returns:
            List of generated scenes
        """
        agent_spec = self.input_processor.process(
            code=code,
            docs=docs,
            spec=spec,
        )

        scenes = self.generator.generate(
            spec=agent_spec,
            mode=mode,
            max_scenes=max_scenes,
        )

        if output_dir:
            self.generator.save_scenes(scenes, Path(output_dir))

        return scenes

    def _generate_scenes_for_plan(
        self,
        toolkit: EvaluatorToolkit,
        spec: AgentSpec,
        plan: IterationPlan,
        include_adversarial: bool,
    ) -> list[Scene]:
        """Generate scenes based on the iteration plan."""
        scenes: list[Scene] = []

        if plan.decision == IterationDecision.DRILL_DOWN:
            for cap_id in plan.target_capability_ids:
                cap = spec.capability_by_id(cap_id)
                if cap:
                    cap_scenes = self.generator.generate_for_capability(
                        spec, cap, include_edge_cases=True
                    )
                    scenes.extend(cap_scenes[:2])

        elif plan.decision == IterationDecision.EXPAND_COVERAGE:
            result = toolkit.generate_scenes(
                mode=GenerationMode.SYSTEMATIC,
                capability_ids=plan.target_capability_ids,
                max_scenes=plan.num_scenes_to_generate,
            )
            if result.success and result.data:
                scenes.extend(result.data)

        elif plan.decision == IterationDecision.ADVERSARIAL_PROBE:
            result = toolkit.generate_scenes(
                mode=GenerationMode.ADVERSARIAL,
                max_scenes=plan.num_scenes_to_generate,
            )
            if result.success and result.data:
                scenes.extend(result.data)

        if include_adversarial and plan.decision != IterationDecision.ADVERSARIAL_PROBE:
            adv_result = toolkit.generate_scenes(
                mode=GenerationMode.ADVERSARIAL,
                max_scenes=2,
            )
            if adv_result.success and adv_result.data:
                scenes.extend(adv_result.data[:2])

        return scenes

    def _build_report(
        self,
        agent_spec: AgentSpec,
        started_at: datetime,
        toolkit: EvaluatorToolkit,
        analysis_results: list[AnalysisResult],
        pattern_analysis: PatternAnalysis | None,
        all_scenes: list[Scene],
        iteration_log: list[dict[str, Any]],
    ) -> EvaluationReport:
        """Build the final evaluation report."""
        finished_at = datetime.now(UTC)
        coverage = toolkit.coverage_tracker.get_report()
        execution_history = toolkit.get_execution_history()

        failures = []
        for result in analysis_results:
            if not result.passed:
                failures.append(
                    FailureReport(
                        scene_id=result.scene_id,
                        capability_id=result.affected_capability,
                        failure_type=result.failure_type or "unknown",
                        description=result.failure_description or "Unknown failure",
                        root_cause=result.root_cause,
                        suggested_fix=result.suggested_fix,
                        severity=result.severity,
                    )
                )

        summary = EvaluationSummary(
            total_scenes=len(execution_history),
            scenes_passed=sum(1 for r in execution_history if r.passed),
            scenes_failed=sum(1 for r in execution_history if not r.passed),
            total_checks=sum(len(r.check_result.checks) for r in execution_history),
            checks_passed=sum(
                sum(1 for c in r.check_result.checks if c.passed)
                for r in execution_history
            ),
            checks_failed=sum(
                sum(1 for c in r.check_result.checks if not c.passed)
                for r in execution_history
            ),
            capabilities_tested=sum(1 for c in coverage.capabilities if c.tested),
            capabilities_covered=sum(1 for c in coverage.capabilities if c.passed),
            constraints_tested=sum(
                1
                for c in coverage.constraints
                if c.compliance_tested or c.violation_tested
            ),
            constraints_violated=sum(
                1
                for c in coverage.constraints
                if c.compliance_tested and not c.compliance_passed
            ),
            iterations_used=len(iteration_log),
            execution_time_seconds=(finished_at - started_at).total_seconds(),
        )

        coverage_dict = {c.capability_id: c.passed for c in coverage.capabilities}
        for c in coverage.constraints:
            coverage_dict[c.constraint_id] = c.compliance_passed

        return EvaluationReport(
            agent_name=agent_spec.name,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
            failures=failures,
            coverage=coverage_dict,
            scenes_generated=[s.id for s in all_scenes],
            iteration_log=iteration_log,
            metadata={
                "model": self.model,
                "simulator_model": self.simulator_model,
                "spec_capabilities": len(agent_spec.capabilities),
                "spec_constraints": len(agent_spec.constraints),
            },
        )
