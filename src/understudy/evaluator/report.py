"""Evaluation reports and summaries."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FailureReport:
    """Report on a single test failure."""

    scene_id: str
    capability_id: str | None
    failure_type: str  # check_failed, exception, timeout, etc.
    description: str
    root_cause: str | None = None
    suggested_fix: str | None = None
    severity: str = "error"  # error, warning, info
    trace_excerpt: str | None = None
    related_failures: list[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    """Summary statistics from an evaluation run."""

    total_scenes: int = 0
    scenes_passed: int = 0
    scenes_failed: int = 0
    total_checks: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    capabilities_tested: int = 0
    capabilities_covered: int = 0
    constraints_tested: int = 0
    constraints_violated: int = 0
    iterations_used: int = 0
    execution_time_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_scenes == 0:
            return 0.0
        return self.scenes_passed / self.total_scenes

    @property
    def coverage_rate(self) -> float:
        if self.capabilities_tested == 0:
            return 0.0
        return self.capabilities_covered / self.capabilities_tested


@dataclass
class EvaluationReport:
    """Complete report from an evaluation run."""

    agent_name: str
    started_at: datetime
    finished_at: datetime | None = None
    summary: EvaluationSummary = field(default_factory=EvaluationSummary)
    failures: list[FailureReport] = field(default_factory=list)
    coverage: dict[str, bool] = field(default_factory=dict)
    scenes_generated: list[str] = field(default_factory=list)
    iteration_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def to_text(self) -> str:
        """Render report as plain text."""
        lines = [
            f"Evaluation Report: {self.agent_name}",
            "=" * 50,
            "",
            "Summary:",
            f"  Pass rate: {self.summary.pass_rate:.1%}",
            f"  Scenes: {self.summary.scenes_passed}/{self.summary.total_scenes} passed",
            f"  Coverage: {self.summary.coverage_rate:.1%}",
            f"  Iterations: {self.summary.iterations_used}",
        ]

        if self.duration_seconds:
            lines.append(f"  Duration: {self.duration_seconds:.1f}s")

        if self.failures:
            lines.append("")
            lines.append(f"Failures ({len(self.failures)}):")
            for f in self.failures:
                lines.append(f"  - [{f.severity}] {f.scene_id}: {f.description}")
                if f.root_cause:
                    lines.append(f"    Root cause: {f.root_cause}")
                if f.suggested_fix:
                    lines.append(f"    Fix: {f.suggested_fix}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "summary": {
                "total_scenes": self.summary.total_scenes,
                "scenes_passed": self.summary.scenes_passed,
                "scenes_failed": self.summary.scenes_failed,
                "pass_rate": self.summary.pass_rate,
                "coverage_rate": self.summary.coverage_rate,
                "iterations_used": self.summary.iterations_used,
            },
            "failures": [
                {
                    "scene_id": f.scene_id,
                    "capability_id": f.capability_id,
                    "failure_type": f.failure_type,
                    "description": f.description,
                    "root_cause": f.root_cause,
                    "suggested_fix": f.suggested_fix,
                    "severity": f.severity,
                }
                for f in self.failures
            ],
            "coverage": self.coverage,
            "scenes_generated": self.scenes_generated,
        }
