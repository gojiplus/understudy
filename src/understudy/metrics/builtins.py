"""Built-in metrics: standard evaluation metrics."""

from ..models import Expectations
from ..trace import Trace
from .registry import MetricRegistry, MetricResult


@MetricRegistry.register("efficiency", description="Token and latency efficiency metrics")
def compute_efficiency(trace: "Trace", expectations: "Expectations") -> MetricResult:
    """Compute efficiency metrics: token counts, latency, turn count."""
    metrics = trace.metrics
    return MetricResult(
        name="efficiency",
        value={
            "turn_count": trace.turn_count,
            "total_input_tokens": metrics.total_input_tokens,
            "total_output_tokens": metrics.total_output_tokens,
            "total_thinking_tokens": metrics.total_thinking_tokens,
            "total_tokens": metrics.total_tokens,
            "agent_time_ms": metrics.agent_time_ms,
            "avg_turn_latency_ms": metrics.avg_turn_latency_ms,
        },
    )


@MetricRegistry.register("resolution_match", description="Check if terminal state matches expected")
def compute_resolution_match(trace: "Trace", expectations: "Expectations") -> MetricResult:
    """Check if the trace terminal_state matches the expected resolution."""
    if not expectations.expected_resolution:
        return MetricResult(
            name="resolution_match",
            passed=True,
            detail="No expected resolution specified",
        )

    actual = trace.terminal_state
    expected = expectations.expected_resolution
    passed = actual == expected
    return MetricResult(
        name="resolution_match",
        passed=passed,
        detail=f"expected={expected}, actual={actual}",
    )


@MetricRegistry.register("tool_trajectory", description="Tool call sequence analysis")
def compute_tool_trajectory(trace: "Trace", expectations: "Expectations") -> MetricResult:
    """Compute tool trajectory metrics: sequence, unique tools, total calls."""
    sequence = trace.call_sequence()
    return MetricResult(
        name="tool_trajectory",
        value={
            "sequence": sequence,
            "unique_tools": list(set(sequence)),
            "total_calls": len(trace.tool_calls),
        },
    )


def _compute_trajectory_score(
    actual: list[str], expected: list[str], mode: str
) -> tuple[float, str]:
    """Compute trajectory match score based on mode."""
    if mode == "exact":
        if actual == expected:
            return 1.0, f"Exact match: {expected}"
        return 0.0, f"Mismatch: expected {expected}, got {actual}"

    if mode == "prefix":
        if actual[: len(expected)] == expected:
            return 1.0, f"Prefix match: {expected}"
        return 0.0, f"Prefix mismatch: expected {expected} at start"

    if mode == "contains":
        it = iter(actual)
        if all(tool in it for tool in expected):
            return 1.0, f"Contains match: {expected} found in order"
        return 0.0, f"Contains mismatch: {expected} not found in order"

    if mode == "subset":
        actual_set = set(actual)
        if all(tool in actual_set for tool in expected):
            return 1.0, f"Subset match: all {expected} called"
        missing = [t for t in expected if t not in actual_set]
        return 0.0, f"Subset mismatch: missing {missing}"

    return 0.0, f"Unknown mode: {mode}"


@MetricRegistry.register("trajectory_match", description="Compare tool sequence against expected")
def compute_trajectory_match(trace: "Trace", expectations: "Expectations") -> MetricResult:
    """Check if actual tool sequence matches expected trajectory."""
    if expectations.expected_trajectory is None:
        return MetricResult(
            name="trajectory_match",
            passed=None,
            detail="No expected trajectory specified",
        )

    actual = trace.call_sequence()
    expected = expectations.expected_trajectory
    mode = expectations.trajectory_match_mode

    score, detail = _compute_trajectory_score(actual, expected, mode)
    passed = score == 1.0

    return MetricResult(
        name="trajectory_match",
        passed=passed,
        value={"expected": expected, "actual": actual, "score": score, "mode": mode},
        detail=detail,
    )
