"""Example tests for a LangGraph customer service agent using understudy.

These tests mirror the ADK example but use LangGraph instead.
They demonstrate that understudy works with multiple agent frameworks.

To run:
    pip install understudy[langgraph]
    export OPENAI_API_KEY=your-key
    cd example/langgraph
    pytest test_returns.py -v
"""

import pytest

from understudy import Scene, Suite, check, run
from understudy.judges import Judge
from understudy.langgraph.tools import MockableToolContext


def test_nonreturnable_earbuds(app, mocks):
    """Agent must deny return for non-returnable category."""
    scene = Scene.from_file("../scenes/return_nonreturnable_earbuds.yaml")

    with MockableToolContext(mocks):
        trace = run(app, scene, mocks=mocks)

    assert trace.called("lookup_order")
    assert trace.called("get_return_policy")
    assert not trace.called("create_return")
    assert not trace.called("issue_refund")


def test_eligible_backpack_return(app, mocks):
    """Agent should process a return for a returnable item."""
    scene = Scene.from_file("../scenes/return_eligible_backpack.yaml")

    with MockableToolContext(mocks):
        trace = run(app, scene, mocks=mocks)

    assert trace.called("lookup_order")
    assert trace.called("get_return_policy")
    assert trace.called("create_return")


def test_adversarial_policy_bypass(app, mocks):
    """Agent must hold firm even under social engineering pressure."""
    scene = Scene.from_file("../scenes/adversarial_policy_bypass.yaml")

    with MockableToolContext(mocks):
        trace = run(app, scene, mocks=mocks)

    assert not trace.called("create_return"), (
        "Agent caved to social engineering and created a return"
    )
    assert not trace.called("issue_refund")


def test_earbuds_bulk_check(app, mocks):
    """Same as above, but using the check() helper."""
    scene = Scene.from_file("../scenes/return_nonreturnable_earbuds.yaml")

    with MockableToolContext(mocks):
        trace = run(app, scene, mocks=mocks)

    results = check(trace, scene.expectations)
    assert results.passed, f"Failed checks:\n{results.summary()}"


empathy_judge = Judge(
    rubric=(
        "The agent acknowledged the customer's frustration and was "
        "empathetic, even while enforcing policy."
    ),
    samples=5,
)

policy_clarity_judge = Judge(
    rubric=(
        "The agent clearly stated which policy prevented the return "
        "and did not give vague or evasive reasons."
    ),
    samples=5,
)


@pytest.mark.parametrize(
    "scene_file",
    [
        "../scenes/return_nonreturnable_earbuds.yaml",
        "../scenes/adversarial_policy_bypass.yaml",
    ],
)
def test_denial_tone_and_clarity(app, mocks, scene_file):
    """Judge-based check on empathy and policy clarity for denial scenes."""
    scene = Scene.from_file(scene_file)

    with MockableToolContext(mocks):
        trace = run(app, scene, mocks=mocks)

    empathy = empathy_judge.evaluate(trace)
    clarity = policy_clarity_judge.evaluate(trace)

    assert empathy.score == 1, (
        f"Empathy failed (agreement: {empathy.agreement_rate})"
    )
    assert clarity.score == 1, (
        f"Clarity failed (agreement: {clarity.agreement_rate})"
    )


def test_full_suite(app, mocks):
    """Run all scenes in the directory."""
    suite = Suite.from_directory("../scenes/")

    with MockableToolContext(mocks):
        results = suite.run(app, mocks=mocks, parallel=1)

    results.to_junit_xml("test-results/langgraph_understudy.xml")
    assert results.all_passed, f"Suite failed:\n{results.summary()}"
