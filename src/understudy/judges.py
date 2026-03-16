"""Judges: calibrated LLM-as-judge with sampling and majority vote."""

from dataclasses import dataclass

from .trace import Trace

FAILURE_ANALYSIS_PROMPT = """\
You are analyzing why an AI agent conversation failed its evaluation.

The conversation was evaluated against these expectations:
{expectations}

The following checks FAILED:
{failed_checks}

Analyze the conversation and explain:
1. What went wrong (be specific, cite the turn or tool call)
2. Root cause (why did the agent make this mistake)
3. How to fix it (actionable suggestion)

Keep your analysis concise (3-5 sentences total).
"""

JUDGE_SYSTEM_PROMPT = """\
You are evaluating the quality of an AI agent's conversation with a user.
You will be given the full conversation transcript including tool calls.

Evaluate ONLY the following criterion:
{rubric}

Respond with exactly one word: YES or NO.
Do not explain your reasoning. Just YES or NO.
"""


@dataclass
class JudgeResult:
    """Result of an LLM judge evaluation."""

    score: int
    raw_scores: list[int]
    agreement_rate: float

    @property
    def unanimous(self) -> bool:
        return self.agreement_rate == 1.0


class Judge:
    """LLM-as-judge with configurable sampling and majority vote.

    Usage::

        judge = Judge(
            rubric="The agent was empathetic throughout.",
            samples=5,
        )
        result = judge.evaluate(trace)
        assert result.score == 1
        assert result.agreement_rate >= 0.6
    """

    def __init__(
        self,
        rubric: str,
        samples: int = 5,
        model: str = "gpt-4o",
    ):
        self.rubric = rubric
        self.samples = samples
        self.model = model

    def evaluate(self, trace: Trace) -> JudgeResult:
        """Evaluate a trace against the rubric using majority vote.

        Calls the judge model `self.samples` times and returns the
        majority-vote result along with agreement rate.
        """
        conversation = trace.conversation_text()
        raw_scores = []

        for _ in range(self.samples):
            score = self._single_eval(conversation)
            raw_scores.append(score)

        yes_count = sum(raw_scores)
        no_count = len(raw_scores) - yes_count
        majority = 1 if yes_count > no_count else 0
        majority_count = yes_count if majority == 1 else no_count
        agreement = majority_count / len(raw_scores)

        return JudgeResult(
            score=majority,
            raw_scores=raw_scores,
            agreement_rate=agreement,
        )

    def _single_eval(self, conversation: str) -> int:
        """Run a single judge evaluation. Returns 1 for YES, 0 for NO."""
        prompt = JUDGE_SYSTEM_PROMPT.format(rubric=self.rubric)
        full_prompt = f"{prompt}\n\nCONVERSATION TRANSCRIPT:\n{conversation}"
        return self._eval_litellm(full_prompt)

    def _eval_litellm(self, prompt: str) -> int:
        """Evaluate using litellm for unified provider access."""
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm package required for LLM judge. "
                "Install with: pip install understudy[judges]"
            ) from e

        response = litellm.completion(
            model=self.model,
            max_tokens=10,
            temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        text = (content or "").strip().upper()
        return 1 if text.startswith("YES") else 0


@dataclass
class FailureAnalysis:
    """Analysis of why a run failed."""

    run_id: str
    scene_id: str
    failed_checks: list[str]
    analysis: str


class FailureAnalyzer:
    """Analyze failed runs using an LLM to identify root causes.

    Usage::

        analyzer = FailureAnalyzer(model="gpt-4o")
        analysis = analyzer.analyze(trace, expectations, failed_checks)
        print(analysis.analysis)
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def analyze(
        self,
        trace: Trace,
        expectations_text: str,
        failed_checks: list[str],
    ) -> str:
        """Analyze why a trace failed and return explanation."""
        conversation = trace.conversation_text()

        prompt = FAILURE_ANALYSIS_PROMPT.format(
            expectations=expectations_text,
            failed_checks="\n".join(f"- {c}" for c in failed_checks),
        )

        full_prompt = f"{prompt}\n\nCONVERSATION TRANSCRIPT:\n{conversation}"

        try:
            import litellm
        except ImportError:
            return "LLM analysis unavailable (litellm not installed)"

        response = litellm.completion(
            model=self.model,
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": full_prompt}],
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
        return (content or "No analysis available").strip()

    def analyze_run(self, run_data: dict) -> FailureAnalysis:
        """Analyze a failed run from storage data."""
        trace: Trace | None = run_data.get("trace")
        scene = run_data.get("scene")
        check = run_data.get("check", {})
        metadata = run_data.get("metadata", {})

        failed_checks = [
            c.get("label", "unknown")
            for c in check.get("checks", [])
            if not c.get("passed")
        ]

        if not failed_checks or not trace:
            return FailureAnalysis(
                run_id=metadata.get("run_id", "unknown"),
                scene_id=metadata.get("scene_id", "unknown"),
                failed_checks=failed_checks,
                analysis="No trace available for analysis." if not trace else "No failures.",
            )

        expectations_text = ""
        if scene:
            exp = scene.expectations
            if exp.required_tools:
                expectations_text += f"Required tools: {exp.required_tools}\n"
            if exp.forbidden_tools:
                expectations_text += f"Forbidden tools: {exp.forbidden_tools}\n"
            if exp.expected_resolution:
                expectations_text += f"Expected resolution: {exp.expected_resolution}\n"

        analysis = self.analyze(trace, expectations_text, failed_checks)

        return FailureAnalysis(
            run_id=metadata.get("run_id", "unknown"),
            scene_id=metadata.get("scene_id", "unknown"),
            failed_checks=failed_checks,
            analysis=analysis,
        )
