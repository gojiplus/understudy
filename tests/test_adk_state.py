"""Actual ADK runtime tests with deterministic agents and models."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("google.adk")
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event, EventActions
from google.adk.models import BaseLlm, LlmResponse
from google.adk.workflow import START, Workflow
from google.genai import types

from understudy import Checkpoint, MockToolkit, StateFixture, run_turn
from understudy.adk import ADKApp


class Counter(BaseAgent):
    async def _run_async_impl(self, ctx):
        n = ctx.session.state.get(self.name, 0) + 1
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={self.name: n}),
            content=types.Content(role="model", parts=[types.Part(text=str(n))]),
        )


def counter_app():
    return ADKApp(Counter(name="counter"), app_id="counter-v1")


def test_four_turn_checkpoint_forks_without_reexecution(tmp_path):
    saved = None
    for _ in range(4):
        result = run_turn(
            counter_app, "increment", checkpoint=saved, capture_checkpoint=True
        )
        saved = result.checkpoint
    assert saved.after_turn == 4
    saved.save(tmp_path / "after4.json")
    loaded = Checkpoint.load(tmp_path / "after4.json")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_turn,
                counter_app,
                "next",
                checkpoint=loaded,
                state_patch={"counter": n},
            )
            for n in [10, 20]
        ]
        results = [f.result() for f in futures]
    assert [r.state_after for r in results] == [{"counter": 11}, {"counter": 21}]
    assert loaded.state == {"counter": 4}
    assert all(len(r.trace.turns) == 2 for r in results)
    assert all(len(r.trace.executions) == 1 for r in results)
    with pytest.raises(ValueError, match="app_id"):
        run_turn(lambda: ADKApp(Counter(name="other")), "hello", checkpoint=saved)


@pytest.mark.parametrize("parallel", [False, True])
def test_real_nested_invocations(parallel):
    def app():
        a, b = Counter(name="a"), Counter(name="b")
        edges = [(START, (a, b))] if parallel else [(START, a), (a, b)]
        return ADKApp(Workflow(name="root", edges=edges))

    result = run_turn(app, "go", capture_checkpoint=True)
    assert result.state_after == {"a": 1, "b": 1}
    spans = result.trace.executions
    root = next(s for s in spans if s.agent_name == "root")
    assert {s.agent_path for s in spans} == {("root",), ("root", "a"), ("root", "b")}
    assert all(s.parent_id == root.id for s in spans if s.agent_name != "root")
    assert all(s.status == "completed" for s in spans)
    assert result.trace.agent_transfers == []


class ToolModel(BaseLlm):
    model: str = "offline"

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1]
        if any(p.function_response for p in last.parts or []):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="done")])
            )
        else:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="write", args={"value": "x"}, id="write-id"
                            )
                        )
                    ],
                )
            )


@pytest.mark.filterwarnings("ignore:.*JSON_SCHEMA_FOR_FUNC_DECL.*:UserWarning")
def test_nested_tool_interception_and_callback_preservation():
    real = []

    def write(value: str) -> dict:
        """Write a value."""
        real.append(value)
        return {"real": value}

    def factory():
        agent = LlmAgent(name="child", model=ToolModel(), tools=[write])
        return ADKApp(Workflow(name="root", edges=[(START, agent)]))

    def mocks():
        toolkit = MockToolkit(unmatched="error")
        toolkit.handle("write", agent_path=("root", "child"))(
            lambda value: {"mock": value}
        )
        return toolkit

    result = run_turn(factory, "write", mocks_factory=mocks)
    assert real == []
    call = result.trace.tool_calls[0]
    assert call.result == {"mock": "x"}
    assert call.agent_path == ("root", "child")
    assert call.execution_id
    assert result.trace.agent_called("child", "write")


def test_synthetic_state():
    result = run_turn(counter_app, "next", fixture=StateFixture(state={"counter": 40}))
    assert result.state_before == {"counter": 40}
    assert result.state_after == {"counter": 41}


def test_tool_history_and_invalid_mock_results():
    from understudy import ToolCall, Turn

    fixture = StateFixture(
        state={"counter": 4},
        history=[
            Turn(role="user", content="old"),
            Turn(
                role="agent",
                content="old answer",
                tool_calls=[ToolCall(tool_name="lookup", result={"value": 1})],
            ),
        ],
    )
    result = run_turn(counter_app, "new", fixture=fixture, capture_checkpoint=True)
    assert result.state_after["counter"] == 5
    assert not result.trace.called("lookup")
    assert any(
        e["content"] and any(p.get("function_response") for p in e["content"]["parts"])
        for e in result.checkpoint.payload["events"]
    )


@pytest.mark.filterwarnings("ignore:.*JSON_SCHEMA_FOR_FUNC_DECL.*:UserWarning")
def test_unmatched_tool_blocked_and_async_mock():
    real = []

    def write(value: str) -> dict:
        """Write a value."""
        real.append(value)
        return {"real": value}

    def app():
        return ADKApp(LlmAgent(name="child", model=ToolModel(), tools=[write]))

    with pytest.raises(Exception, match="Unmocked tool"):
        run_turn(app, "write", mocks_factory=lambda: MockToolkit(unmatched="error"))
    assert real == []

    def mocks():
        toolkit = MockToolkit()

        async def mocked(value):
            return {"async": value}

        toolkit.handle("write")(mocked)
        return toolkit

    result = run_turn(app, "write", mocks_factory=mocks)
    assert result.trace.tool_calls[0].result == {"async": "x"}
    assert real == []
    result = run_turn(app, "write")
    assert real == ["x"]
    assert result.trace.tool_calls[0].result == {"real": "x"}


@pytest.mark.filterwarnings("ignore:.*JSON_SCHEMA_FOR_FUNC_DECL.*:UserWarning")
def test_parallel_agents_can_reuse_function_call_ids():
    class RepeatedIdModel(BaseLlm):
        model: str = "offline"
        calls: int = 0

        async def generate_content_async(self, llm_request, stream=False):
            self.calls += 1
            part = (
                types.Part(
                    function_call=types.FunctionCall(
                        name="write", args={"value": "x"}, id="same-id"
                    )
                )
                if self.calls == 1
                else types.Part(text="done")
            )
            yield LlmResponse(content=types.Content(role="model", parts=[part]))

    def write(value: str) -> dict:
        """Write a value."""
        pytest.fail("mocked tools must not execute")

    def app():
        a = LlmAgent(name="a", model=RepeatedIdModel(), tools=[write])
        b = LlmAgent(name="b", model=RepeatedIdModel(), tools=[write])
        sink = Counter(name="sink")
        return ADKApp(Workflow(name="root", edges=[(START, (a, b)), ((a, b), sink)]))

    def mocks():
        toolkit = MockToolkit(unmatched="error")
        toolkit.handle("write", agent_path=("root", "a"))(lambda value: {"agent": "a"})
        toolkit.handle("write", agent_path=("root", "b"))(lambda value: {"agent": "b"})
        return toolkit

    result = run_turn(app, "go", mocks_factory=mocks)
    calls = result.trace.tool_calls
    assert len(calls) == 2
    assert all(c.result == {"agent": c.agent_name} for c in calls)
    assert len({c.execution_id for c in calls}) == 2


def test_pending_tools_cannot_be_checkpointed():
    from understudy import UnsupportedCapabilityError

    class Waiting(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="approval", id="pending"
                            )
                        )
                    ]
                ),
            )

    def factory():
        return ADKApp(Waiting(name="waiting"))

    result = run_turn(factory, "go")
    assert result.trace.terminal_state == "interrupted"
    with pytest.raises(UnsupportedCapabilityError, match="pending"):
        run_turn(factory, "go", capture_checkpoint=True)
