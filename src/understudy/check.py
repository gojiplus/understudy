"""Check: validate a trace against scene expectations."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import MetricRegistry, MetricResult
from .models import Expectations
from .trace import Trace


@dataclass
class CheckResult:
    """Result of checking a trace against expectations."""

    checks: list["CheckItem"] = field(default_factory=list)
    metrics: dict[str, MetricResult] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        checks_passed = all(c.passed for c in self.checks)
        metrics_passed = all(m.passed for m in self.metrics.values() if m.passed is not None)
        return checks_passed and metrics_passed

    @property
    def failed_checks(self) -> list["CheckItem"]:
        return [c for c in self.checks if not c.passed]

    @property
    def failed_metrics(self) -> list[MetricResult]:
        return [m for m in self.metrics.values() if m.passed is False]

    def metric(self, name: str) -> MetricResult | None:
        return self.metrics.get(name)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            mark = "✓" if c.passed else "✗"
            lines.append(f"  {mark} {c.label}: {c.detail}")
        for name, m in self.metrics.items():
            if m.passed is not None:
                mark = "✓" if m.passed else "✗"
                lines.append(f"  {mark} {name}: {m.detail}")
            else:
                lines.append(f"  • {name}: {m.value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_checks = len(self.checks)
        n_pass = sum(1 for c in self.checks if c.passed)
        n_metrics = len(self.metrics)
        if n_metrics:
            return f"CheckResult({n_pass}/{n_checks} checks, {n_metrics} metrics)"
        return f"CheckResult({n_pass}/{n_checks} passed)"


@dataclass
class CheckItem:
    """A single check result."""

    label: str
    passed: bool
    detail: str


def check(trace: Trace, expectations: Expectations) -> CheckResult:
    """Validate a trace against expectations.

    Args:
        trace: The execution trace from a rehearsal.
        expectations: The expectations from a scene.

    Returns:
        A CheckResult with individual check outcomes.
    """
    result = CheckResult()
    called_tools = set(trace.call_sequence())

    # required tools
    for tool in expectations.required_tools:
        result.checks.append(
            CheckItem(
                label="required_tool",
                passed=tool in called_tools,
                detail=f"{tool} {'called' if tool in called_tools else 'NOT called'}",
            )
        )

    # forbidden tools
    for tool in expectations.forbidden_tools:
        was_called = tool in called_tools
        result.checks.append(
            CheckItem(
                label="forbidden_tool",
                passed=not was_called,
                detail=f"{tool} {'CALLED (violation)' if was_called else 'not called'}",
            )
        )

    # required agents
    invoked_agents = set(trace.agents_invoked())
    for agent in expectations.required_agents:
        result.checks.append(
            CheckItem(
                label="required_agent",
                passed=agent in invoked_agents,
                detail=f"{agent} {'invoked' if agent in invoked_agents else 'NOT invoked'}",
            )
        )

    # forbidden agents
    for agent in expectations.forbidden_agents:
        was_invoked = agent in invoked_agents
        result.checks.append(
            CheckItem(
                label="forbidden_agent",
                passed=not was_invoked,
                detail=f"{agent} {'INVOKED (violation)' if was_invoked else 'not invoked'}",
            )
        )

    # required agent tools
    for agent, tools in expectations.required_agent_tools.items():
        for tool in tools:
            called = trace.agent_called(agent, tool)
            result.checks.append(
                CheckItem(
                    label="required_agent_tool",
                    passed=called,
                    detail=f"{agent}.{tool} {'called' if called else 'NOT called'}",
                )
            )

    # expected resolution check
    if expectations.expected_resolution:
        passed = trace.terminal_state == expectations.expected_resolution
        expected = expectations.expected_resolution
        actual = trace.terminal_state
        result.checks.append(
            CheckItem(
                label="expected_resolution",
                passed=passed,
                detail=f"expected={expected}, actual={actual}",
            )
        )

    # compute metrics
    if expectations.metrics:
        result.metrics = MetricRegistry.compute_all(expectations.metrics, trace, expectations)

    return result


def evaluate(
    trace: Trace,
    expectations: Expectations,
    metrics: list[str] | None = None,
    judge_model: str | None = None,
    judges: dict[str, Any] | None = None,
) -> CheckResult:
    """Evaluate a trace against expectations.

    This is an alias for check() with clearer naming for the evaluate workflow.

    Args:
        trace: The execution trace to evaluate.
        expectations: The expectations to check against.
        metrics: Override metrics to compute (defaults to scene's metrics).
        judge_model: Optional LLM model for judge evaluations.
        judges: Optional pre-configured judge objects.

    Returns:
        A CheckResult with check outcomes and metrics.
    """
    if metrics:
        expectations = Expectations(
            required_tools=expectations.required_tools,
            forbidden_tools=expectations.forbidden_tools,
            required_agents=expectations.required_agents,
            forbidden_agents=expectations.forbidden_agents,
            required_agent_tools=expectations.required_agent_tools,
            expected_resolution=expectations.expected_resolution,
            metrics=metrics,
        )

    result = check(trace, expectations)

    if judge_model and judges:
        for name, judge in judges.items():
            judge_result = judge.evaluate(trace)
            result.metrics[f"judge_{name}"] = MetricResult(
                name=f"judge_{name}",
                value={"score": judge_result.score, "agreement_rate": judge_result.agreement_rate},
                passed=judge_result.score == 1,
                detail=judge_result.reasoning or "",
            )

    return result


@dataclass
class EvaluationResult:
    """Result of evaluating a single trace."""

    trace_id: str
    check_result: CheckResult
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.check_result.passed


def evaluate_batch(
    traces: list[Trace] | str | Path,
    expectations: Expectations | None = None,
    output: str | Path | None = None,
    judge_model: str | None = None,
    metrics: list[str] | None = None,
    parallel: int = 1,
) -> list[EvaluationResult]:
    """Evaluate multiple traces and return results.

    Args:
        traces: List of Trace objects, or path to trace file/directory.
        expectations: Expectations to evaluate against (if None, loads from trace metadata).
        output: Optional path to save evaluation results.
        judge_model: Optional LLM model for judge evaluations.
        metrics: Override metrics to compute.
        parallel: Number of parallel evaluation threads.

    Returns:
        List of EvaluationResult objects.
    """
    from .storage import EvaluationStorage, TraceStorage

    trace_list: list[tuple[str, Trace, Expectations]] = []

    if isinstance(traces, (str, Path)):
        path = Path(traces)
        storage = TraceStorage(path=path)
        for trace_id in storage.list_traces():
            data = storage.load(trace_id)
            trace = data["trace"]
            scene = data["scene"]
            exp = expectations if expectations else scene.expectations
            trace_list.append((trace_id, trace, exp))
    else:
        for i, trace in enumerate(traces):
            trace_id = f"trace_{i}"
            exp = expectations if expectations else Expectations()
            trace_list.append((trace_id, trace, exp))

    result_storage = None
    if output:
        result_storage = EvaluationStorage(path=Path(output))

    results: list[EvaluationResult] = []

    def evaluate_single(trace_id: str, trace: Trace, exp: Expectations) -> EvaluationResult:
        try:
            check_result = evaluate(
                trace=trace,
                expectations=exp,
                metrics=metrics,
                judge_model=judge_model,
            )
            if result_storage:
                result_storage.save(trace_id=trace_id, check_result=check_result)
            return EvaluationResult(trace_id=trace_id, check_result=check_result)
        except Exception as e:
            return EvaluationResult(
                trace_id=trace_id,
                check_result=CheckResult(),
                error=str(e),
            )

    if parallel <= 1:
        for trace_id, trace, exp in trace_list:
            result = evaluate_single(trace_id, trace, exp)
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(evaluate_single, trace_id, trace, exp): trace_id
                for trace_id, trace, exp in trace_list
            }
            for future in as_completed(futures):
                results.append(future.result())

    return results
