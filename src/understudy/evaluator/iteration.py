"""Iteration controller for the evaluator agent.

Decides when to drill down on failures, expand coverage,
refine tests, or stop.
"""

from dataclasses import dataclass
from enum import StrEnum

from .analyzer import AnalysisResult, PatternAnalysis
from .coverage import CoverageReport


class IterationDecision(StrEnum):
    """What the evaluator should do next."""

    DRILL_DOWN = "drill_down"
    EXPAND_COVERAGE = "expand_coverage"
    REFINE_TESTS = "refine_tests"
    ADVERSARIAL_PROBE = "adversarial_probe"
    STOP = "stop"


@dataclass
class IterationState:
    """Current state of the evaluation iteration."""

    iteration_number: int
    max_iterations: int
    scenes_run: int
    scenes_passed: int
    scenes_failed: int
    coverage_rate: float
    patterns_found: int
    budget_remaining: float  # 0-1, how much budget remains


@dataclass
class IterationPlan:
    """Plan for the next iteration."""

    decision: IterationDecision
    reason: str
    target_capability_ids: list[str]
    target_constraint_ids: list[str]
    num_scenes_to_generate: int
    generation_mode: str  # systematic, adversarial, focused
    focus_areas: list[str]


class IterationController:
    """Controls the evaluation iteration loop.

    Decides based on current state what the evaluator should do next:
    - Drill down on failures to understand root causes
    - Expand coverage to test untested capabilities
    - Refine tests that are flaky or unclear
    - Probe adversarially to find edge cases
    - Stop when sufficient coverage is achieved
    """

    def __init__(
        self,
        max_iterations: int = 3,
        min_coverage: float = 0.8,
        max_failure_rate: float = 0.3,
        adversarial_after_coverage: float = 0.7,
    ):
        """Initialize the controller.

        Args:
            max_iterations: Maximum number of iterations
            min_coverage: Minimum coverage rate to consider "done"
            max_failure_rate: If failure rate exceeds this, focus on fixing
            adversarial_after_coverage: Coverage threshold to start adversarial
        """
        self.max_iterations = max_iterations
        self.min_coverage = min_coverage
        self.max_failure_rate = max_failure_rate
        self.adversarial_after_coverage = adversarial_after_coverage

    def decide_next(
        self,
        state: IterationState,
        coverage: CoverageReport,
        analysis_results: list[AnalysisResult],
        pattern_analysis: PatternAnalysis | None = None,
    ) -> IterationPlan:
        """Decide what to do next.

        Args:
            state: Current iteration state
            coverage: Current coverage report
            analysis_results: Analysis of current failures
            pattern_analysis: Optional pattern analysis

        Returns:
            Plan for next iteration
        """
        if state.iteration_number >= state.max_iterations:
            return IterationPlan(
                decision=IterationDecision.STOP,
                reason="Maximum iterations reached",
                target_capability_ids=[],
                target_constraint_ids=[],
                num_scenes_to_generate=0,
                generation_mode="none",
                focus_areas=[],
            )

        if state.budget_remaining <= 0:
            return IterationPlan(
                decision=IterationDecision.STOP,
                reason="Budget exhausted",
                target_capability_ids=[],
                target_constraint_ids=[],
                num_scenes_to_generate=0,
                generation_mode="none",
                focus_areas=[],
            )

        failure_rate = state.scenes_failed / max(state.scenes_run, 1)
        if failure_rate > self.max_failure_rate:
            return self._plan_drill_down(coverage, analysis_results, pattern_analysis)

        if coverage.overall_coverage_rate < self.min_coverage:
            return self._plan_coverage_expansion(coverage)

        if coverage.overall_coverage_rate >= self.min_coverage and failure_rate <= 0.1:
            return IterationPlan(
                decision=IterationDecision.STOP,
                reason=(
                    f"Coverage target reached ({coverage.overall_coverage_rate:.1%}) "
                    f"with low failure rate ({failure_rate:.1%})"
                ),
                target_capability_ids=[],
                target_constraint_ids=[],
                num_scenes_to_generate=0,
                generation_mode="none",
                focus_areas=[],
            )

        if coverage.overall_coverage_rate >= self.adversarial_after_coverage:
            return self._plan_adversarial(coverage, analysis_results)

        return self._plan_coverage_expansion(coverage)

    def _plan_drill_down(
        self,
        coverage: CoverageReport,
        analysis_results: list[AnalysisResult],
        pattern_analysis: PatternAnalysis | None,
    ) -> IterationPlan:
        """Plan to drill down on failures."""
        failing_caps = []
        failing_cons = []
        focus_areas = []

        for result in analysis_results:
            if not result.passed:
                if result.affected_capability:
                    failing_caps.append(result.affected_capability)
                if result.affected_constraint:
                    failing_cons.append(result.affected_constraint)
                if result.root_cause:
                    focus_areas.append(result.root_cause)

        if pattern_analysis and pattern_analysis.patterns:
            for pattern in pattern_analysis.patterns[:3]:
                failing_caps.extend(pattern.affected_capabilities)
                if pattern.root_cause_hypothesis:
                    focus_areas.append(pattern.root_cause_hypothesis)

        failing_caps = list(set(failing_caps))[:5]
        failing_cons = list(set(failing_cons))[:3]
        focus_areas = list(set(focus_areas))[:5]

        num_scenes = min(5, max(2, len(failing_caps) + len(failing_cons)))

        return IterationPlan(
            decision=IterationDecision.DRILL_DOWN,
            reason=f"High failure rate - investigating {len(failing_caps)} capabilities",
            target_capability_ids=failing_caps,
            target_constraint_ids=failing_cons,
            num_scenes_to_generate=num_scenes,
            generation_mode="focused",
            focus_areas=focus_areas,
        )

    def _plan_coverage_expansion(
        self,
        coverage: CoverageReport,
    ) -> IterationPlan:
        """Plan to expand test coverage."""
        untested_caps = [c.capability_id for c in coverage.untested_capabilities]
        untested_cons = [c.constraint_id for c in coverage.untested_constraints]

        target_caps = untested_caps[:5]
        target_cons = untested_cons[:3]

        num_scenes = min(10, len(target_caps) + len(target_cons) * 2)
        num_scenes = max(3, num_scenes)

        return IterationPlan(
            decision=IterationDecision.EXPAND_COVERAGE,
            reason=(
                f"Coverage at {coverage.overall_coverage_rate:.1%} "
                f"- testing {len(target_caps)} capabilities"
            ),
            target_capability_ids=target_caps,
            target_constraint_ids=target_cons,
            num_scenes_to_generate=num_scenes,
            generation_mode="systematic",
            focus_areas=[],
        )

    def _plan_adversarial(
        self,
        coverage: CoverageReport,
        analysis_results: list[AnalysisResult],
    ) -> IterationPlan:
        """Plan adversarial probing."""
        weak_caps = []
        for result in analysis_results:
            if not result.passed and result.affected_capability:
                weak_caps.append(result.affected_capability)

        for cap in coverage.failing_capabilities:
            if cap.capability_id not in weak_caps:
                weak_caps.append(cap.capability_id)

        weak_caps = list(set(weak_caps))[:5]

        return IterationPlan(
            decision=IterationDecision.ADVERSARIAL_PROBE,
            reason=f"Good coverage ({coverage.overall_coverage_rate:.1%}) - probing edge cases",
            target_capability_ids=weak_caps,
            target_constraint_ids=[],
            num_scenes_to_generate=5,
            generation_mode="adversarial",
            focus_areas=["boundary conditions", "policy bypass", "unusual inputs"],
        )

    def get_iteration_state(
        self,
        iteration_number: int,
        scenes_run: int,
        scenes_passed: int,
        coverage: CoverageReport,
        patterns_found: int = 0,
    ) -> IterationState:
        """Build current iteration state.

        Args:
            iteration_number: Current iteration (0-indexed)
            scenes_run: Total scenes run so far
            scenes_passed: Total scenes passed so far
            coverage: Current coverage report
            patterns_found: Number of failure patterns found

        Returns:
            Current iteration state
        """
        budget_used = iteration_number / max(self.max_iterations, 1)
        return IterationState(
            iteration_number=iteration_number,
            max_iterations=self.max_iterations,
            scenes_run=scenes_run,
            scenes_passed=scenes_passed,
            scenes_failed=scenes_run - scenes_passed,
            coverage_rate=coverage.overall_coverage_rate,
            patterns_found=patterns_found,
            budget_remaining=1.0 - budget_used,
        )
