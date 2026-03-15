"""Built-in metrics: standard evaluation metrics."""

from .registry import MetricRegistry, MetricResult

if True:
    from ..models import Expectations
    from ..trace import Trace


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
