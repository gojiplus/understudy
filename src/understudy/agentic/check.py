"""Check: validate an agentic trace against expectations."""

import fnmatch
from dataclasses import dataclass, field
from typing import Any

from .models import AgenticExpectations, AgenticTrace


@dataclass
class AgenticCheckItem:
    """A single check result for agentic evaluation."""

    label: str
    passed: bool
    detail: str


@dataclass
class AgenticCheckResult:
    """Result of checking an agentic trace against expectations."""

    checks: list[AgenticCheckItem] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[AgenticCheckItem]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            mark = "+" if c.passed else "-"
            lines.append(f"  {mark} {c.label}: {c.detail}")
        for name, value in self.metrics.items():
            lines.append(f"  * {name}: {value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_checks = len(self.checks)
        n_pass = sum(1 for c in self.checks if c.passed)
        return f"AgenticCheckResult({n_pass}/{n_checks} passed)"


def _matches_pattern(action: str, pattern: str) -> bool:
    """Check if an action matches a pattern (supports wildcards)."""
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatch(action, pattern)
    return action == pattern


def _evaluate_predicate(predicate: str, trace: AgenticTrace) -> bool:
    """Evaluate a goal predicate against the trace.

    Supports:
    - Simple expressions: "outcome == 'success'"
    - Access to trace fields: outcome, total_steps, total_tokens, final_state
    """
    local_vars = {
        "outcome": trace.outcome,
        "total_steps": trace.total_steps,
        "total_tokens": trace.total_tokens,
        "final_state": trace.final_state,
        "artifacts": [a.model_dump() for a in trace.artifacts],
        "steps": trace.steps,
    }
    try:
        result = eval(predicate, {"__builtins__": {}}, local_vars)
        return bool(result)
    except Exception:
        return False


def _compare_outputs(actual: dict, expected: dict) -> tuple[bool, str]:
    """Compare actual output against golden output."""
    differences = []

    for key, expected_value in expected.items():
        if key not in actual:
            differences.append(f"missing key: {key}")
        elif actual[key] != expected_value:
            differences.append(f"{key}: expected {expected_value}, got {actual[key]}")

    if differences:
        return False, "; ".join(differences)
    return True, "output matches expected"


def check_agentic(trace: AgenticTrace, expectations: AgenticExpectations) -> AgenticCheckResult:
    """Validate an agentic trace against expectations.

    Args:
        trace: The agentic execution trace.
        expectations: The expectations to check against.

    Returns:
        An AgenticCheckResult with individual check outcomes.
    """
    result = AgenticCheckResult()
    performed_actions = set(trace.action_sequence())

    if expectations.goal_predicate:
        predicate_passed = _evaluate_predicate(expectations.goal_predicate, trace)
        result.checks.append(
            AgenticCheckItem(
                label="goal_predicate",
                passed=predicate_passed,
                detail=(
                    f"predicate '{expectations.goal_predicate}' "
                    f"{'satisfied' if predicate_passed else 'NOT satisfied'}"
                ),
            )
        )

    if expectations.golden_output:
        output_passed, output_detail = _compare_outputs(
            trace.final_state, expectations.golden_output
        )
        result.checks.append(
            AgenticCheckItem(
                label="golden_output",
                passed=output_passed,
                detail=output_detail,
            )
        )

    if expectations.max_steps is not None:
        steps_ok = trace.total_steps <= expectations.max_steps
        result.checks.append(
            AgenticCheckItem(
                label="max_steps",
                passed=steps_ok,
                detail=(
                    f"steps={trace.total_steps}, max={expectations.max_steps} "
                    f"({'OK' if steps_ok else 'EXCEEDED'})"
                ),
            )
        )

    if expectations.max_tokens is not None:
        tokens_ok = trace.total_tokens <= expectations.max_tokens
        result.checks.append(
            AgenticCheckItem(
                label="max_tokens",
                passed=tokens_ok,
                detail=(
                    f"tokens={trace.total_tokens}, max={expectations.max_tokens} "
                    f"({'OK' if tokens_ok else 'EXCEEDED'})"
                ),
            )
        )

    if expectations.max_retries is not None:
        retries = trace.retry_count()
        retries_ok = retries <= expectations.max_retries
        result.checks.append(
            AgenticCheckItem(
                label="max_retries",
                passed=retries_ok,
                detail=(
                    f"retries={retries}, max={expectations.max_retries} "
                    f"({'OK' if retries_ok else 'EXCEEDED'})"
                ),
            )
        )

    for action in expectations.required_actions:
        found = any(_matches_pattern(a, action) for a in performed_actions)
        result.checks.append(
            AgenticCheckItem(
                label="required_action",
                passed=found,
                detail=f"{action} {'performed' if found else 'NOT performed'}",
            )
        )

    for pattern in expectations.forbidden_actions:
        violations = [a for a in performed_actions if _matches_pattern(a, pattern)]
        passed = len(violations) == 0
        result.checks.append(
            AgenticCheckItem(
                label="forbidden_action",
                passed=passed,
                detail=f"{pattern} VIOLATED by {violations}" if violations else f"{pattern} OK",
            )
        )

    result.metrics["outcome"] = trace.outcome
    result.metrics["total_steps"] = trace.total_steps
    result.metrics["total_tokens"] = trace.total_tokens
    result.metrics["total_latency_ms"] = trace.total_latency_ms
    result.metrics["retry_count"] = trace.retry_count()

    return result
