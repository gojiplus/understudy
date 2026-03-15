"""Judges: calibrated LLM-as-judge with sampling and majority vote."""

from dataclasses import dataclass

from .trace import Trace

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
