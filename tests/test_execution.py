"""Shared adapter contracts and single-turn boundaries."""

from copy import deepcopy

import pytest

from understudy import (
    AdapterCapabilities,
    AgentResponse,
    Checkpoint,
    MockToolkit,
    StateFixture,
    ToolCall,
    Trace,
    Turn,
    UnsupportedCapabilityError,
    replay_to,
    run_turn,
)


class CounterApp:
    capabilities = AdapterCapabilities(
        tools=True, agents=True, state=True, checkpoints=True, interception=True
    )

    def __init__(self):
        self.closed = False
        self.sent = []
        self.state = {}

    def start(self, mocks=None):
        self.mocks = mocks

    def inject(self, fixture):
        self.state = deepcopy(fixture.state)

    def restore(self, checkpoint):
        self.state = deepcopy(checkpoint.state)

    def validate_checkpoint(self, checkpoint):
        if checkpoint.adapter != "counter":
            raise ValueError("wrong adapter")

    def patch_state(self, patch):
        self.state.update(patch)

    def get_state(self):
        return self.state

    def send(self, message):
        self.sent.append(message)
        self.state["n"] = self.state.get("n", 0) + 1
        if self.mocks:
            self.mocks.state["n"] = self.mocks.state.get("n", 0) + 1
        return AgentResponse(
            str(self.state["n"]), tool_calls=[ToolCall(tool_name="new")]
        )

    def checkpoint(self):
        return Checkpoint(
            adapter="counter",
            framework_version="1",
            app_id="counter",
            state=deepcopy(self.state),
        )

    def stop(self):
        self.closed = True


def test_next_turn_isolated_from_history_and_source(tmp_path):
    fixture = StateFixture(
        state={"n": 4},
        history=[
            Turn(role="user", content="old"),
            Turn(role="agent", content="old", tool_calls=[ToolCall(tool_name="old")]),
        ],
        tool_state={"n": 10},
    )
    first = run_turn(
        CounterApp,
        "new",
        fixture=fixture,
        mocks_factory=MockToolkit,
        capture_checkpoint=True,
    )
    assert first.trace.called("new")
    assert not first.trace.called("old")
    assert first.state_before == {"n": 4}
    assert first.state_after == {"n": 5}
    assert fixture.state == {"n": 4}
    assert first.state_changes["n"]["before"] == 4
    first.checkpoint.save(tmp_path / "fixture.json")
    saved = Checkpoint.load(tmp_path / "fixture.json")
    assert saved.tool_state == {"n": 11}
    a = run_turn(
        CounterApp,
        "a",
        checkpoint=saved,
        state_patch={"n": 20},
        mocks_factory=MockToolkit,
        capture_checkpoint=True,
    )
    b = run_turn(
        CounterApp,
        "b",
        checkpoint=saved,
        mocks_factory=MockToolkit,
        capture_checkpoint=True,
    )
    assert a.state_after == {"n": 21}
    assert b.state_after == {"n": 6}
    assert b.checkpoint.tool_state == {"n": 12}
    assert saved.state == {"n": 5}


def test_generate_one_message_and_close_on_failure():
    calls = []

    def generate(history):
        calls.append(history)
        history.clear()
        return "next"

    source = StateFixture(
        history=[Turn(role="user", content="old"), Turn(role="agent", content="old")]
    )
    result = run_turn(CounterApp, fixture=source, next_message=generate)
    assert len(calls) == 1
    assert len(source.history) == 2
    assert len(result.trace.turns) == 2
    app = CounterApp()
    with pytest.raises(TypeError, match="string"):
        run_turn(lambda: app, next_message=lambda _: None)
    assert app.closed
    with pytest.raises(ValueError, match="exactly one"):
        run_turn(CounterApp)
    with pytest.raises(ValueError, match="exactly one"):
        run_turn(CounterApp, "hello", next_message=generate)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixture": StateFixture()},
        {"capture_checkpoint": True},
        {"state_patch": {}},
        {"mocks_factory": MockToolkit},
    ],
)
def test_unsupported_operations_rejected_before_execution(kwargs):
    app = CounterApp()
    app.capabilities = AdapterCapabilities()
    with pytest.raises(UnsupportedCapabilityError):
        run_turn(lambda: app, "hello", **kwargs)
    assert app.sent == []
    assert app.closed


def test_unknown_observations_cannot_pass_negative_assertions():
    app = CounterApp()
    app.capabilities = AdapterCapabilities()
    result = run_turn(lambda: app, "hello")
    with pytest.raises(ValueError, match="observe tool"):
        result.trace.called("danger")
    with pytest.raises(ValueError, match="agent identity"):
        result.trace.agents_invoked()


def test_replay_to_boundary_and_divergence():
    trace = Trace(
        scene_id="prefix",
        turns=[
            Turn(role="user", content="first"),
            Turn(role="agent", content="1", tool_calls=[ToolCall(tool_name="new")]),
            Turn(role="user", content="second"),
            Turn(role="agent", content="2", tool_calls=[ToolCall(tool_name="new")]),
        ],
    )
    checkpoint = replay_to(CounterApp, trace, after_turn=1)
    assert checkpoint.after_turn == 1
    assert checkpoint.state == {"n": 1}
    assert len(checkpoint.history) == 2
    assert run_turn(CounterApp, "next", checkpoint=checkpoint).state_after == {"n": 2}
    assert replay_to(CounterApp, trace, after_turn=0).state == {}
    trace.turns[1].content = "different"
    with pytest.raises(ValueError, match="diverged at turn 1"):
        replay_to(CounterApp, trace, after_turn=1)
    assert replay_to(CounterApp, trace, after_turn=1, require_match=False).state == {
        "n": 1
    }
    with pytest.raises(ValueError, match="outside"):
        replay_to(CounterApp, trace, after_turn=3)


def test_scoped_mocks_and_strict_policy():
    mocks = MockToolkit(unmatched="error")
    mocks.handle("lookup")(lambda: "global")
    mocks.handle("lookup", agent_path=("root", "billing"))(lambda: "billing")
    assert mocks.resolve("lookup", ("root", "billing", "child"))() == "billing"
    assert mocks.resolve("lookup", ("root", "other"))() == "global"
    with pytest.raises(Exception, match="Unmocked tool"):
        mocks.resolve("write", ("root",))


def test_parallel_suites_and_batches_use_fresh_apps_and_mocks():
    from understudy import Persona, Scene, Suite, simulate_batch

    class EndingApp(CounterApp):
        def send(self, message):
            self.mocks.state["n"] = self.mocks.state.get("n", 0) + 1
            return AgentResponse(str(self.mocks.state["n"]), terminal_state="done")

    scene = Scene(
        id="parallel",
        starting_prompt="hello",
        conversation_plan="",
        persona=Persona(description="test"),
    )
    traces = simulate_batch(
        EndingApp, [scene], n_sims=4, parallel=2, mocks_factory=MockToolkit
    )
    assert [t.turns[-1].content for t in traces] == ["1"] * 4
    results = Suite([scene]).run(
        EndingApp, n_sims=4, parallel=2, mocks_factory=MockToolkit
    )
    assert [r.trace.turns[-1].content for r in results.results] == ["1"] * 4
    with pytest.raises(ValueError, match="app factory"):
        simulate_batch(EndingApp(), [scene], parallel=2)
    with pytest.raises(ValueError, match="mocks_factory"):
        simulate_batch(EndingApp, [scene], parallel=2, mocks=MockToolkit())
    with pytest.raises(ValueError, match="app factory"):
        Suite([scene]).run(EndingApp(), parallel=2)
    with pytest.raises(ValueError, match="mocks_factory"):
        Suite([scene]).run(EndingApp, parallel=2, mocks=MockToolkit())


def test_initialization_errors_close_without_executing():
    app = CounterApp()
    saved = app.checkpoint()
    with pytest.raises(ValueError, match="not both"):
        run_turn(lambda: app, "hello", checkpoint=saved, fixture=StateFixture())
    assert app.closed
    with pytest.raises(ValueError, match="complete alternating"):
        StateFixture(history=[Turn(role="user", content="unanswered")])
    from understudy.http import HTTPApp

    app = HTTPApp("https://example.test", "test")
    with pytest.raises(UnsupportedCapabilityError, match="cannot be intercepted"):
        app.start(MockToolkit())
