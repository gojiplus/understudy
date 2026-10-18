"""Exercise ADK's event contract without calling a model."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from understudy import MockToolkit
from understudy.adk import ADKApp, _load_dotenv

adk = pytest.importorskip("google.adk")
events = pytest.importorskip("google.adk.events")
types = pytest.importorskip("google.genai.types")


@pytest.fixture
def session(monkeypatch):
    service = Mock()
    service.create_session = AsyncMock(
        return_value=SimpleNamespace(id="session", state={"order": "pending"})
    )
    service.get_session = AsyncMock(
        side_effect=lambda **kw: service.create_session.return_value
    )
    runner = Mock()
    runner.close = AsyncMock()
    monkeypatch.setattr("google.adk.apps.App", Mock())
    monkeypatch.setattr(adk, "Runner", Mock(return_value=runner))
    monkeypatch.setattr(
        "google.adk.sessions.InMemorySessionService", Mock(return_value=service)
    )
    return runner, service


def stream_events(runner, *items):
    async def stream(**kwargs):
        for item in items:
            yield item

    runner.run_async = Mock(side_effect=stream)


@pytest.mark.parametrize("session_id", [None, "requested-session"])
def test_lifecycle(session, session_id):
    runner, service = session
    agent = SimpleNamespace(name="assistant", before_tool_callback=None)
    app = ADKApp(agent, session_id=session_id)
    app.start()
    arguments = service.create_session.call_args.kwargs
    assert arguments["app_name"] == "understudy_test"
    assert arguments["user_id"] == "understudy_user"
    assert arguments["session_id"]
    if session_id:
        assert arguments["session_id"] == session_id
    stream_events(runner, events.Event(author="assistant"))
    response = app.send("hello")
    assert response.content == ""
    assert response.state_snapshot == {"order": "pending"}
    assert runner.run_async.call_args.kwargs["new_message"].parts[0].text == "hello"
    app.stop()
    app.stop()
    with pytest.raises(RuntimeError, match="session is not open"):
        app.send("after stop")


def test_event_trace(session):
    runner, _ = session
    app = ADKApp(SimpleNamespace(name="assistant"))
    app.start()
    stream_events(
        runner,
        events.Event(
            author="assistant",
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="lookup", args={"id": 1})
                    ),
                    types.Part(function_call=types.FunctionCall(name="lookup")),
                    types.Part(function_call=types.FunctionCall()),
                ]
            ),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=2, thoughts_token_count=3
            ),
        ),
        events.Event(
            author="assistant",
            content=types.Content(
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="unknown", response={}
                        )
                    ),
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="lookup", response={"first": True}
                        )
                    ),
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="lookup", response={"second": True}
                        )
                    ),
                ]
            ),
            actions=events.EventActions(transfer_to_agent="specialist"),
        ),
        events.Event(
            author="specialist",
            content=types.Content(parts=[types.Part(text="Done"), types.Part()]),
            actions=events.EventActions(escalate=True),
            usage_metadata=types.GenerateContentResponseUsageMetadata(),
        ),
    )
    response = app.send("look up my orders")
    assert response.content == "Done"
    assert [call.arguments for call in response.tool_calls] == [{"id": 1}, {}]
    assert [call.result for call in response.tool_calls] == [
        {"first": True},
        {"second": True},
    ]
    assert response.agent_name == "specialist"
    assert [(t.from_agent, t.to_agent) for t in response.agent_transfers] == [
        ("assistant", "specialist")
    ]
    assert (
        response.input_tokens,
        response.output_tokens,
        response.thinking_tokens,
    ) == (10, 2, 3)
    assert response.terminal_state == "agent_ended"
    app.stop()


def test_transfers_are_per_turn(session):
    runner, _ = session
    app = ADKApp(SimpleNamespace(name="assistant"))
    app.start()
    stream_events(
        runner,
        events.Event(
            author="assistant",
            actions=events.EventActions(transfer_to_agent="specialist"),
        ),
    )
    first = app.send("transfer")
    assert len(first.agent_transfers) == 1
    stream_events(runner, events.Event(author="specialist"))
    second = app.send("continue")
    assert second.agent_transfers == []
    assert len(first.agent_transfers) == 1
    app.stop()


def test_empty_author_and_repeated_transfer(session):
    runner, service = session
    service.create_session.return_value.state = {}
    app = ADKApp(SimpleNamespace())
    app.start()
    stream_events(
        runner,
        events.Event(actions=events.EventActions(transfer_to_agent="first")),
        events.Event(
            author="first", actions=events.EventActions(transfer_to_agent="first")
        ),
    )
    response = app.send("hello")
    assert response.agent_name == "first"
    assert response.agent_transfers == []
    assert response.state_snapshot == {}
    app.stop()


def test_callbacks_are_not_replaced(session):
    original = Mock()
    agent = SimpleNamespace(name="assistant", before_tool_callback=original)
    app = ADKApp(agent)
    app.start(MockToolkit())
    assert agent.before_tool_callback is original
    app.stop()
    assert agent.before_tool_callback is original


@pytest.mark.parametrize(
    ("module", "operation"), [("google.adk", "start"), ("google.genai", "send")]
)
def test_missing_dependencies(monkeypatch, module, operation):
    monkeypatch.setitem(sys.modules, module, None)
    app = ADKApp(SimpleNamespace(name="assistant"))
    action = (
        app.start
        if operation == "start"
        else lambda: asyncio.run(app._send_async("hello"))
    )
    with pytest.raises(ImportError, match=r"understudy\[adk\]"):
        action()


def test_dotenv_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)
    _load_dotenv()
