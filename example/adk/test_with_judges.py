"""Full evaluation with all LLM judge rubrics.

Run with:
    cd example/adk
    pytest test_with_judges.py -v

Requires ANTHROPIC_API_KEY (or other provider key) for judges.
"""

import pytest

from understudy import (
    ADVERSARIAL_ROBUSTNESS,
    FACTUAL_GROUNDING,
    INSTRUCTION_FOLLOWING,
    POLICY_COMPLIANCE,
    TASK_COMPLETION,
    TONE_EMPATHY,
    TOOL_USAGE_CORRECTNESS,
    Judge,
    Scene,
    check,
    run,
)

RUBRICS = [
    ("tool_usage", TOOL_USAGE_CORRECTNESS),
    ("policy_compliance", POLICY_COMPLIANCE),
    ("tone_empathy", TONE_EMPATHY),
    ("adversarial_robustness", ADVERSARIAL_ROBUSTNESS),
    ("task_completion", TASK_COMPLETION),
    ("factual_grounding", FACTUAL_GROUNDING),
    ("instruction_following", INSTRUCTION_FOLLOWING),
]


class TestNonreturnableEarbuds:
    """Tests for the non-returnable earbuds denial scenario."""

    @pytest.fixture
    def trace(self, app, mocks):
        scene = Scene.from_file("../scenes/return_nonreturnable_earbuds.yaml")
        return run(app, scene, mocks=mocks)

    def test_deterministic_checks(self, trace):
        scene = Scene.from_file("../scenes/return_nonreturnable_earbuds.yaml")
        results = check(trace, scene.expectations)
        assert results.passed, f"Failed:\n{results.summary()}"

    @pytest.mark.parametrize("rubric_name,rubric", RUBRICS)
    def test_judge_rubric(self, trace, rubric_name, rubric):
        judge = Judge(rubric=rubric, samples=5)
        result = judge.evaluate(trace)
        assert result.score == 1, (
            f"{rubric_name} failed (agreement: {result.agreement_rate})"
        )


class TestEligibleBackpack:
    """Tests for the eligible backpack return scenario."""

    @pytest.fixture
    def trace(self, app, mocks):
        scene = Scene.from_file("../scenes/return_eligible_backpack.yaml")
        return run(app, scene, mocks=mocks)

    def test_deterministic_checks(self, trace):
        scene = Scene.from_file("../scenes/return_eligible_backpack.yaml")
        results = check(trace, scene.expectations)
        assert results.passed, f"Failed:\n{results.summary()}"

    @pytest.mark.parametrize("rubric_name,rubric", RUBRICS)
    def test_judge_rubric(self, trace, rubric_name, rubric):
        judge = Judge(rubric=rubric, samples=5)
        result = judge.evaluate(trace)
        assert result.score == 1, (
            f"{rubric_name} failed (agreement: {result.agreement_rate})"
        )


class TestAdversarialPolicyBypass:
    """Tests for the adversarial social engineering scenario."""

    @pytest.fixture
    def trace(self, app, mocks):
        scene = Scene.from_file("../scenes/adversarial_policy_bypass.yaml")
        return run(app, scene, mocks=mocks)

    def test_deterministic_checks(self, trace):
        scene = Scene.from_file("../scenes/adversarial_policy_bypass.yaml")
        results = check(trace, scene.expectations)
        assert results.passed, f"Failed:\n{results.summary()}"

    @pytest.mark.parametrize("rubric_name,rubric", RUBRICS)
    def test_judge_rubric(self, trace, rubric_name, rubric):
        judge = Judge(rubric=rubric, samples=5)
        result = judge.evaluate(trace)
        assert result.score == 1, (
            f"{rubric_name} failed (agreement: {result.agreement_rate})"
        )
