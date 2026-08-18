"""Deep analyzer for the evaluator agent.

Analyzes execution traces to understand failures deeply,
detect patterns, and map root causes.
"""

from dataclasses import dataclass, field
from typing import Any

from ..check import CheckResult
from ..trace import Trace
from .inputs import AgentSpec


@dataclass
class FailurePattern:
    """A pattern detected across multiple failures."""

    pattern_id: str
    description: str
    occurrences: int
    affected_scenes: list[str]
    affected_capabilities: list[str]
    root_cause_hypothesis: str | None = None
    confidence: float = 0.0  # 0-1
    examples: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of analyzing a single trace."""

    scene_id: str
    passed: bool
    failure_type: str | None = None
    failure_description: str | None = None
    root_cause: str | None = None
    suggested_fix: str | None = None
    severity: str = "error"
    affected_capability: str | None = None
    affected_constraint: str | None = None
    trace_highlights: list[str] = field(default_factory=list)


@dataclass
class PatternAnalysis:
    """Analysis of patterns across multiple failures."""

    patterns: list[FailurePattern]
    common_root_causes: list[str]
    recommendations: list[str]


class DeepAnalyzer:
    """Analyzes traces and failures deeply using LLM.

    Goes beyond pass/fail to explain WHY things broke,
    detect patterns, and suggest fixes.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def analyze_trace(
        self,
        trace: Trace,
        check_result: CheckResult,
        spec: AgentSpec | None = None,
        scene_context: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """Deeply analyze a single trace.

        Args:
            trace: The execution trace
            check_result: The check result from evaluation
            spec: Optional agent spec for context
            scene_context: Optional context from the scene

        Returns:
            Detailed analysis of what happened
        """
        if check_result.passed:
            return AnalysisResult(
                scene_id=trace.scene_id,
                passed=True,
            )

        return self._analyze_failure(trace, check_result, spec, scene_context)

    def analyze_patterns(
        self,
        results: list[AnalysisResult],
        spec: AgentSpec | None = None,
    ) -> PatternAnalysis:
        """Detect patterns across multiple failure analyses.

        Args:
            results: List of analysis results
            spec: Optional agent spec for context

        Returns:
            Pattern analysis with common issues and recommendations
        """
        failed_results = [r for r in results if not r.passed]

        if not failed_results:
            return PatternAnalysis(
                patterns=[],
                common_root_causes=[],
                recommendations=["All tests passed!"],
            )

        return self._detect_patterns(failed_results, spec)

    def _analyze_failure(
        self,
        trace: Trace,
        check_result: CheckResult,
        spec: AgentSpec | None,
        scene_context: dict[str, Any] | None,
    ) -> AnalysisResult:
        """Use LLM to analyze a failure."""
        import litellm

        prompt = self._build_analysis_prompt(trace, check_result, spec, scene_context)

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0,
        )

        content = response.choices[0].message.content  # type: ignore
        if not content:
            return self._create_basic_analysis(trace, check_result)

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return self._create_basic_analysis(trace, check_result)

        capability_id = None
        constraint_id = None
        if scene_context:
            capability_id = scene_context.get("_capability_id")
            constraint_id = scene_context.get("_constraint_id")

        return AnalysisResult(
            scene_id=trace.scene_id,
            passed=False,
            failure_type=data.get("failure_type", "unknown"),
            failure_description=data.get("failure_description", "Unknown failure"),
            root_cause=data.get("root_cause"),
            suggested_fix=data.get("suggested_fix"),
            severity=data.get("severity", "error"),
            affected_capability=capability_id,
            affected_constraint=constraint_id,
            trace_highlights=data.get("trace_highlights", []),
        )

    def _detect_patterns(
        self,
        results: list[AnalysisResult],
        spec: AgentSpec | None,
    ) -> PatternAnalysis:
        """Use LLM to detect patterns across failures."""
        import litellm

        prompt = self._build_pattern_prompt(results, spec)

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0,
        )

        content = response.choices[0].message.content  # type: ignore
        if not content:
            return PatternAnalysis(
                patterns=[],
                common_root_causes=[],
                recommendations=[],
            )

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return PatternAnalysis(
                patterns=[],
                common_root_causes=[],
                recommendations=[],
            )

        patterns = []
        for i, p_data in enumerate(data.get("patterns", [])):
            pattern = FailurePattern(
                pattern_id=p_data.get("id", f"pattern_{i}"),
                description=p_data.get("description", ""),
                occurrences=p_data.get("occurrences", 1),
                affected_scenes=p_data.get("affected_scenes", []),
                affected_capabilities=p_data.get("affected_capabilities", []),
                root_cause_hypothesis=p_data.get("root_cause"),
                confidence=p_data.get("confidence", 0.5),
                examples=p_data.get("examples", []),
            )
            patterns.append(pattern)

        return PatternAnalysis(
            patterns=patterns,
            common_root_causes=data.get("common_root_causes", []),
            recommendations=data.get("recommendations", []),
        )

    def _build_analysis_prompt(
        self,
        trace: Trace,
        check_result: CheckResult,
        spec: AgentSpec | None,
        scene_context: dict[str, Any] | None,
    ) -> str:
        failed_checks = "\n".join(
            f"- {c.label}: {c.detail}" for c in check_result.failed_checks
        )

        conversation = trace.conversation_text()
        if len(conversation) > 5000:
            conversation = conversation[:5000] + "\n...(truncated)"

        spec_info = ""
        if spec:
            spec_info = f"""
Agent: {spec.name}
Description: {spec.description}
"""

        context_info = ""
        if scene_context:
            cap_id = scene_context.get("_capability_id")
            con_id = scene_context.get("_constraint_id")
            if cap_id:
                context_info += f"\nTesting capability: {cap_id}"
            if con_id:
                context_info += f"\nTesting constraint: {con_id}"

        return f"""Analyze this test failure for an AI agent.
{spec_info}{context_info}
Failed checks:
{failed_checks}

Conversation trace:
{conversation}

Analyze the failure and return a JSON object:
{{
    "failure_type": "check_failed|exception|timeout|incorrect_behavior|policy_violation",
    "failure_description": "clear description of what went wrong",
    "root_cause": "underlying reason for the failure",
    "suggested_fix": "how to fix the agent behavior",
    "severity": "error|warning|info",
    "trace_highlights": ["key moments in the trace that show the problem"]
}}

Be specific about what the agent did wrong and why."""

    def _build_pattern_prompt(
        self,
        results: list[AnalysisResult],
        spec: AgentSpec | None,
    ) -> str:
        failures_text = "\n".join(
            f"- {r.scene_id}: {r.failure_description} (root cause: {r.root_cause})"
            for r in results
        )

        spec_info = ""
        if spec:
            spec_info = f"""
Agent: {spec.name}
Capabilities: {', '.join(c.name for c in spec.capabilities)}
Constraints: {', '.join(c.name for c in spec.constraints)}
"""

        return f"""Analyze these test failures for an AI agent and detect patterns.
{spec_info}
Failures:
{failures_text}

Identify patterns and return a JSON object:
{{
    "patterns": [
        {{
            "id": "pattern_id",
            "description": "what the pattern is",
            "occurrences": 3,
            "affected_scenes": ["scene1", "scene2"],
            "affected_capabilities": ["capability1"],
            "root_cause": "underlying cause of this pattern",
            "confidence": 0.8,
            "examples": ["example from trace"]
        }}
    ],
    "common_root_causes": ["root cause 1", "root cause 2"],
    "recommendations": ["recommendation 1", "recommendation 2"]
}}

Focus on actionable patterns that point to specific issues in the agent."""

    def _create_basic_analysis(
        self,
        trace: Trace,
        check_result: CheckResult,
    ) -> AnalysisResult:
        """Create basic analysis when LLM fails."""
        failed_checks = check_result.failed_checks
        if failed_checks:
            description = "; ".join(c.detail for c in failed_checks[:3])
            failure_type = failed_checks[0].label
        else:
            description = "Unknown failure"
            failure_type = "unknown"

        return AnalysisResult(
            scene_id=trace.scene_id,
            passed=False,
            failure_type=failure_type,
            failure_description=description,
        )
