"""Metrics: agentic-specific evaluation metrics."""

from dataclasses import dataclass
from typing import Any

from .models import AgenticExpectations, AgenticTrace


@dataclass
class AgenticMetricResult:
    """Result of computing an agentic metric."""

    name: str
    value: Any
    passed: bool | None = None
    detail: str = ""

    def __repr__(self) -> str:
        if self.passed is not None:
            status = "passed" if self.passed else "failed"
            return f"AgenticMetricResult({self.name}: {status})"
        return f"AgenticMetricResult({self.name}: {self.value})"


def goal_completion(trace: AgenticTrace) -> AgenticMetricResult:
    """Compute goal completion metric based on outcome.

    Returns 1.0 for success, 0.0 for failure/error.
    """
    success_outcomes = {"success", "completed", "done", "finished"}
    completed = trace.outcome.lower() in success_outcomes

    return AgenticMetricResult(
        name="goal_completion",
        value=1.0 if completed else 0.0,
        passed=completed,
        detail=f"outcome='{trace.outcome}'",
    )


def reasoning_quality(trace: AgenticTrace) -> AgenticMetricResult:
    """Analyze the quality of reasoning steps.

    Computes basic metrics about the reasoning process:
    - Number of thinking steps
    - Average reasoning length
    - Think-before-act ratio
    """
    thinking_steps = trace.thinking_steps()
    action_steps = [s for s in trace.steps if s.step_type == "act"]

    n_thinking = len(thinking_steps)
    n_actions = len(action_steps)

    if n_thinking == 0:
        return AgenticMetricResult(
            name="reasoning_quality",
            value={
                "thinking_steps": 0,
                "action_steps": n_actions,
                "think_act_ratio": 0.0,
                "avg_reasoning_length": 0,
            },
            detail="No thinking steps recorded",
        )

    total_reasoning_length = sum(len(s.reasoning or "") for s in thinking_steps)
    avg_reasoning_length = total_reasoning_length / n_thinking

    think_act_ratio = n_thinking / max(n_actions, 1)

    return AgenticMetricResult(
        name="reasoning_quality",
        value={
            "thinking_steps": n_thinking,
            "action_steps": n_actions,
            "think_act_ratio": round(think_act_ratio, 2),
            "avg_reasoning_length": int(avg_reasoning_length),
        },
        detail=f"{n_thinking} thinking steps, ratio={think_act_ratio:.2f}",
    )


def action_efficiency(
    trace: AgenticTrace, expectations: AgenticExpectations | None = None
) -> AgenticMetricResult:
    """Compute action efficiency metrics.

    Includes:
    - Steps per action (lower is better)
    - Retry rate (lower is better)
    - Token efficiency
    """
    total_steps = trace.total_steps
    action_steps = [s for s in trace.steps if s.step_type == "act"]
    n_actions = len(action_steps)
    n_retries = trace.retry_count()
    total_tokens = trace.total_tokens

    if n_actions == 0:
        return AgenticMetricResult(
            name="action_efficiency",
            value={
                "total_steps": total_steps,
                "action_steps": 0,
                "retry_count": 0,
                "retry_rate": 0.0,
                "tokens_per_action": 0,
            },
            detail="No actions performed",
        )

    retry_rate = n_retries / n_actions
    tokens_per_action = total_tokens / n_actions if total_tokens > 0 else 0

    efficiency_passed = None
    if expectations:
        checks = []
        if expectations.max_steps is not None:
            checks.append(total_steps <= expectations.max_steps)
        if expectations.max_retries is not None:
            checks.append(n_retries <= expectations.max_retries)
        if expectations.max_tokens is not None:
            checks.append(total_tokens <= expectations.max_tokens)
        if checks:
            efficiency_passed = all(checks)

    return AgenticMetricResult(
        name="action_efficiency",
        value={
            "total_steps": total_steps,
            "action_steps": n_actions,
            "retry_count": n_retries,
            "retry_rate": round(retry_rate, 3),
            "tokens_per_action": int(tokens_per_action),
            "total_tokens": total_tokens,
            "total_latency_ms": trace.total_latency_ms,
        },
        passed=efficiency_passed,
        detail=f"{n_actions} actions, {n_retries} retries, {total_tokens} tokens",
    )


def action_trajectory(trace: AgenticTrace) -> AgenticMetricResult:
    """Compute action trajectory metrics."""
    sequence = trace.action_sequence()
    unique_actions = list(set(sequence))

    return AgenticMetricResult(
        name="action_trajectory",
        value={
            "sequence": sequence,
            "unique_actions": unique_actions,
            "total_actions": len(sequence),
            "unique_count": len(unique_actions),
        },
        detail=f"{len(sequence)} actions, {len(unique_actions)} unique",
    )


def compute_all_metrics(
    trace: AgenticTrace, expectations: AgenticExpectations | None = None
) -> dict[str, AgenticMetricResult]:
    """Compute all agentic metrics."""
    return {
        "goal_completion": goal_completion(trace),
        "reasoning_quality": reasoning_quality(trace),
        "action_efficiency": action_efficiency(trace, expectations),
        "action_trajectory": action_trajectory(trace),
    }
