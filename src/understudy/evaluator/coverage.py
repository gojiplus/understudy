"""Coverage tracking for the evaluator agent.

Tracks what capabilities have been tested and identifies gaps.
"""

from dataclasses import dataclass, field
from typing import Any

from ..models import Scene
from ..trace import Trace
from .inputs import AgentSpec, Capability, Constraint


@dataclass
class CapabilityCoverage:
    """Coverage status for a single capability."""

    capability_id: str
    capability_name: str
    tested: bool = False
    passed: bool = False
    scenes_tested: list[str] = field(default_factory=list)
    edge_cases_tested: int = 0
    total_edge_cases: int = 0


@dataclass
class ConstraintCoverage:
    """Coverage status for a single constraint."""

    constraint_id: str
    constraint_name: str
    compliance_tested: bool = False
    violation_tested: bool = False
    compliance_passed: bool = False
    violation_passed: bool = False
    scenes_tested: list[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Full coverage report."""

    capabilities: list[CapabilityCoverage]
    constraints: list[ConstraintCoverage]
    total_scenes_run: int = 0
    scenes_passed: int = 0

    @property
    def capability_coverage_rate(self) -> float:
        if not self.capabilities:
            return 1.0
        tested = sum(1 for c in self.capabilities if c.tested)
        return tested / len(self.capabilities)

    @property
    def constraint_coverage_rate(self) -> float:
        if not self.constraints:
            return 1.0
        tested = sum(
            1 for c in self.constraints if c.compliance_tested or c.violation_tested
        )
        return tested / len(self.constraints)

    @property
    def overall_coverage_rate(self) -> float:
        total = len(self.capabilities) + len(self.constraints)
        if total == 0:
            return 1.0

        covered_caps = sum(1 for c in self.capabilities if c.tested)
        covered_cons = sum(
            1 for c in self.constraints if c.compliance_tested or c.violation_tested
        )
        return (covered_caps + covered_cons) / total

    @property
    def untested_capabilities(self) -> list[CapabilityCoverage]:
        return [c for c in self.capabilities if not c.tested]

    @property
    def untested_constraints(self) -> list[ConstraintCoverage]:
        return [
            c
            for c in self.constraints
            if not c.compliance_tested and not c.violation_tested
        ]

    @property
    def failing_capabilities(self) -> list[CapabilityCoverage]:
        return [c for c in self.capabilities if c.tested and not c.passed]

    def to_text(self) -> str:
        lines = [
            "Coverage Report",
            "=" * 40,
            f"Overall coverage: {self.overall_coverage_rate:.1%}",
            f"Capability coverage: {self.capability_coverage_rate:.1%}",
            f"Constraint coverage: {self.constraint_coverage_rate:.1%}",
            f"Scenes run: {self.total_scenes_run}",
            f"Scenes passed: {self.scenes_passed}",
            "",
        ]

        if self.capabilities:
            lines.append("Capabilities:")
            for cap in self.capabilities:
                status = (
                    "PASS" if cap.passed else ("FAIL" if cap.tested else "NOT TESTED")
                )
                lines.append(f"  [{status}] {cap.capability_name}")
                if cap.tested:
                    lines.append(f"    Scenes: {', '.join(cap.scenes_tested)}")
                    edge_status = f"{cap.edge_cases_tested}/{cap.total_edge_cases}"
                    lines.append(f"    Edge cases: {edge_status}")

        if self.constraints:
            lines.append("")
            lines.append("Constraints:")
            for con in self.constraints:
                compliance = (
                    "PASS"
                    if con.compliance_passed
                    else ("FAIL" if con.compliance_tested else "-")
                )
                violation = (
                    "PASS"
                    if con.violation_passed
                    else ("FAIL" if con.violation_tested else "-")
                )
                lines.append(
                    f"  {con.constraint_name}: compliance={compliance}, violation={violation}"
                )

        return "\n".join(lines)


class CoverageTracker:
    """Tracks test coverage of agent capabilities.

    Maps scenes and traces to capabilities and constraints
    to understand what has been tested.
    """

    def __init__(self, spec: AgentSpec):
        self.spec = spec
        self._capability_coverage: dict[str, CapabilityCoverage] = {}
        self._constraint_coverage: dict[str, ConstraintCoverage] = {}
        self._scenes_run: list[str] = []
        self._scenes_passed: list[str] = []

        for cap in spec.capabilities:
            self._capability_coverage[cap.id] = CapabilityCoverage(
                capability_id=cap.id,
                capability_name=cap.name,
                total_edge_cases=len(cap.edge_cases),
            )

        for con in spec.constraints:
            self._constraint_coverage[con.id] = ConstraintCoverage(
                constraint_id=con.id,
                constraint_name=con.name,
            )

    def record_scene(
        self,
        scene: Scene,
        trace: Trace,
        passed: bool,
    ) -> None:
        """Record that a scene was run.

        Args:
            scene: The scene that was run
            trace: The execution trace
            passed: Whether the scene passed
        """
        self._scenes_run.append(scene.id)
        if passed:
            self._scenes_passed.append(scene.id)

        capability_id = scene.context.get("_capability_id")
        if capability_id and capability_id in self._capability_coverage:
            cov = self._capability_coverage[capability_id]
            cov.tested = True
            cov.scenes_tested.append(scene.id)
            if passed:
                cov.passed = True

            if "edge_case" in scene.id:
                cov.edge_cases_tested += 1

        constraint_id = scene.context.get("_constraint_id")
        if constraint_id and constraint_id in self._constraint_coverage:
            cov = self._constraint_coverage[constraint_id]
            cov.scenes_tested.append(scene.id)

            if "compliance" in scene.id:
                cov.compliance_tested = True
                if passed:
                    cov.compliance_passed = True
            elif "violation" in scene.id:
                cov.violation_tested = True
                if passed:
                    cov.violation_passed = True

        self._infer_coverage_from_tools(scene, trace, passed)

    def _infer_coverage_from_tools(
        self,
        scene: Scene,
        trace: Trace,
        passed: bool,
    ) -> None:
        """Infer coverage from tools used in trace."""
        tools_called = set(trace.call_sequence())

        for cap_id, cov in self._capability_coverage.items():
            cap = self.spec.capability_by_id(cap_id)
            if not cap:
                continue

            if cap.tools_involved:
                tools_used = tools_called & set(cap.tools_involved)
                if tools_used and not cov.tested:
                    cov.tested = True
                    cov.scenes_tested.append(scene.id)
                    if passed:
                        cov.passed = True

    def get_report(self) -> CoverageReport:
        """Get current coverage report."""
        return CoverageReport(
            capabilities=list(self._capability_coverage.values()),
            constraints=list(self._constraint_coverage.values()),
            total_scenes_run=len(self._scenes_run),
            scenes_passed=len(self._scenes_passed),
        )

    def get_untested_capabilities(self) -> list[Capability]:
        """Get capabilities that haven't been tested."""
        untested = []
        for cap_id, cov in self._capability_coverage.items():
            if not cov.tested:
                cap = self.spec.capability_by_id(cap_id)
                if cap:
                    untested.append(cap)
        return untested

    def get_untested_constraints(self) -> list[Constraint]:
        """Get constraints that haven't been tested."""
        untested = []
        for con_id, cov in self._constraint_coverage.items():
            if not cov.compliance_tested and not cov.violation_tested:
                for con in self.spec.constraints:
                    if con.id == con_id:
                        untested.append(con)
                        break
        return untested

    def get_failing_capabilities(self) -> list[Capability]:
        """Get capabilities that failed their tests."""
        failing = []
        for cap_id, cov in self._capability_coverage.items():
            if cov.tested and not cov.passed:
                cap = self.spec.capability_by_id(cap_id)
                if cap:
                    failing.append(cap)
        return failing

    def suggest_next_tests(self, max_suggestions: int = 5) -> list[dict[str, Any]]:
        """Suggest what should be tested next.

        Returns list of suggestions with:
        - type: "capability" or "constraint"
        - id: capability/constraint id
        - reason: why this should be tested
        - priority: 1 (highest) to 5 (lowest)
        """
        suggestions = []

        for cap in self.get_failing_capabilities():
            suggestions.append(
                {
                    "type": "capability",
                    "id": cap.id,
                    "name": cap.name,
                    "reason": "Previous test failed - needs investigation",
                    "priority": 1,
                }
            )

        for cap in self.get_untested_capabilities():
            suggestions.append(
                {
                    "type": "capability",
                    "id": cap.id,
                    "name": cap.name,
                    "reason": "Not yet tested",
                    "priority": cap.priority + 1,
                }
            )

        for con in self.get_untested_constraints():
            suggestions.append(
                {
                    "type": "constraint",
                    "id": con.id,
                    "name": con.name,
                    "reason": "Constraint not tested",
                    "priority": 2 if con.severity == "error" else 3,
                }
            )

        for cap_id, cov in self._capability_coverage.items():
            cap = self.spec.capability_by_id(cap_id)
            if cap and cov.edge_cases_tested < cov.total_edge_cases:
                suggestions.append(
                    {
                        "type": "capability",
                        "id": cap.id,
                        "name": cap.name,
                        "reason": (
                            f"Edge cases: {cov.edge_cases_tested}/"
                            f"{cov.total_edge_cases} tested"
                        ),
                        "priority": 4,
                    }
                )

        suggestions.sort(key=lambda s: s["priority"])
        return suggestions[:max_suggestions]
