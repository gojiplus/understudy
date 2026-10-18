"""HTTP adapter contracts using in-memory transports."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from understudy.http import AgentEngineApp, HTTPApp

httpx = pytest.importorskip("httpx")


def test_tool_responses_without_ids_and_private_thoughts():
    from understudy.http import _parse_events

    event = {
        "content": {
            "parts": [
                {"text": "private", "thought": True},
                {"functionCall": {"args": {}}},
                {"functionCall": {"name": "lookup"}},
                {"functionResponse": {"name": "unknown", "response": {}}},
                {
                    "functionResponse": {
                        "id": "unknown",
                        "name": "lookup",
                        "response": {},
                    }
                },
                {"functionResponse": {"name": "lookup", "response": {"found": True}}},
                {"functionResponse": {"name": "lookup", "response": {"ignored": True}}},
            ]
        }
    }
    response = _parse_events([json.dumps(event)], None)
    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == {}
    assert response.tool_calls[0].result == {"found": True}


@pytest.fixture
def transport(monkeypatch):
    requests = []
    replies = []
    clients = []
    client_type = httpx.Client

    def handle(request):
        requests.append(request)
        return replies.pop(0)

    def client(**kwargs):
        instance = client_type(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(httpx, "Client", client)
    return requests, replies, clients


def make_app(engine):
    if engine:
        auth = pytest.importorskip("google.auth.credentials")
        credentials = SimpleNamespace(
            token="test-token",  # noqa: S106
            token_state=auth.TokenState.FRESH,
        )
        return AgentEngineApp(
            "project", "us-central1", "resource", credentials=credentials
        )
    return HTTPApp("https://agent.example/", "assistant", headers={"X-Test": "yes"})


@pytest.mark.parametrize("engine", [False, True])
def test_session_and_request_contract(transport, engine):
    requests, replies, clients = transport
    replies.append(httpx.Response(200, json={"id": "server-session"}))
    app = make_app(engine)
    app.start()
    assert len(requests) == 1
    request = requests[0]
    if engine:
        assert request.url.path.endswith("reasoningEngines/resource:query")
        assert json.loads(request.content) == {
            "class_method": "async_create_session",
            "input": {"user_id": "understudy_user"},
        }
        assert request.headers["Authorization"] == "Bearer test-token"
    else:
        assert "/apps/assistant/users/understudy_user/sessions/" in request.url.path
        assert request.headers["X-Test"] == "yes"
    replies.append(httpx.Response(200, text=""))
    response = app.send("hello")
    payload = json.loads(requests[1].content)
    if engine:
        assert requests[1].url.path.endswith(":streamQuery")
        assert payload == {
            "class_method": "async_stream_query",
            "input": {
                "user_id": "understudy_user",
                "session_id": "server-session",
                "message": "hello",
            },
        }
    else:
        assert requests[1].url.path == "/run_sse"
        assert payload == {
            "app_name": "assistant",
            "user_id": "understudy_user",
            "session_id": "server-session",
            "new_message": {"role": "user", "parts": [{"text": "hello"}]},
        }
    assert response.content == ""
    app.stop()
    app.stop()
    assert clients[0].is_closed
    with pytest.raises(RuntimeError, match="not started"):
        app.send("after stop")


@pytest.mark.parametrize("engine", [False, True])
def test_real_adk_events(transport, engine):
    events = pytest.importorskip("google.adk.events")
    types = pytest.importorskip("google.genai.types")
    requests, replies, _ = transport
    replies.append(httpx.Response(200, json={"id": "session"}))
    app = make_app(engine)
    app.start()
    event_list = [
        events.Event(
            author="assistant",
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="lookup", args={"id": 1}, id="one"
                        )
                    ),
                    types.Part(
                        function_call=types.FunctionCall(
                            name="lookup", args={"id": 2}, id="two"
                        )
                    ),
                ]
            ),
        ),
        events.Event(
            author="assistant",
            content=types.Content(
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="lookup", response={"value": 2}, id="two"
                        )
                    ),
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="lookup", response={"value": 1}, id="one"
                        )
                    ),
                ]
            ),
            actions=events.EventActions(transfer_to_agent="specialist"),
        ),
        events.Event(
            author="specialist",
            content=types.Content(parts=[types.Part(text="Done")]),
            actions=events.EventActions(escalate=True),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10, candidates_token_count=2, thoughts_token_count=3
            ),
        ),
    ]
    lines = [
        event.model_dump_json(by_alias=not engine, exclude_none=True)
        for event in event_list
    ]
    body = "\n".join(lines if engine else [f"data: {line}\n" for line in lines])
    replies.append(httpx.Response(200, text=body))
    response = app.send("lookup")
    assert response.content == "Done"
    assert [call.arguments for call in response.tool_calls] == [{"id": 1}, {"id": 2}]
    assert [call.result for call in response.tool_calls] == [{"value": 1}, {"value": 2}]
    assert [(t.from_agent, t.to_agent) for t in response.agent_transfers] == [
        ("assistant", "specialist")
    ]
    assert response.agent_name == "specialist"
    assert response.terminal_state == "agent_ended"
    assert (
        response.input_tokens,
        response.output_tokens,
        response.thinking_tokens,
    ) == (10, 2, 3)
    replies.append(
        httpx.Response(
            200,
            text=body.splitlines()[-1] if engine else 'data: {"author":"specialist"}\n',
        )
    )
    assert app.send("continue").agent_transfers == []
    assert len(requests) == 3
    app.stop()


@pytest.mark.parametrize("engine", [False, True])
def test_empty_malformed_events_and_implicit_transfer(transport, engine):
    _, replies, _ = transport
    replies.append(httpx.Response(200, json={"id": "session"}))
    app = make_app(engine)
    app.start()
    items = [
        "",
        ": heartbeat",
        "data:",
        "broken json",
        "null",
        "[]",
        "{}",
        '{"author":"first", "content":null}',
        '{"author":"second", "actions":null, "content":{"parts":null}}',
    ]
    body = "\n".join(items if engine else [f"data: {item}" for item in items])
    replies.append(httpx.Response(200, text=body))
    response = app.send("hello")
    assert response.content == ""
    assert response.tool_calls == []
    assert response.terminal_state is None
    assert response.agent_transfers == []
    app.stop()


@pytest.mark.parametrize("engine", [False, True])
def test_request_errors_and_cleanup(transport, engine):
    _, replies, clients = transport
    app = make_app(engine)
    replies.append(httpx.Response(503))
    with pytest.raises(RuntimeError, match="503"):
        app.start()
    assert clients[0].is_closed
    replies.append(httpx.Response(200, json={"id": "session"}))
    app.start()
    replies.append(httpx.Response(403))
    with pytest.raises(RuntimeError, match="403"):
        app.send("denied")
    app.stop()
    assert clients[1].is_closed


def test_http_dependency_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError, match=r"understudy\[http\]"):
        HTTPApp("https://agent.example", "app").start()


@pytest.mark.parametrize("supplied", [False, True])
def test_credentials_refresh(monkeypatch, supplied):
    auth = pytest.importorskip("google.auth")
    credentials = SimpleNamespace(
        token=None, token_state=auth.credentials.TokenState.INVALID
    )
    credentials.refresh = Mock(
        side_effect=lambda request: setattr(credentials, "token", "refreshed")
    )
    default = Mock(return_value=(credentials, "project"))
    monkeypatch.setattr(auth, "default", default)
    app = AgentEngineApp("p", "l", "r", credentials=credentials if supplied else None)
    assert app._get_access_token() == "refreshed"
    credentials.refresh.assert_called_once()
    assert default.call_count == (0 if supplied else 1)


def test_credentials_errors(monkeypatch):
    auth = pytest.importorskip("google.auth")
    monkeypatch.setattr(auth, "default", Mock(side_effect=ValueError("no credentials")))
    app = AgentEngineApp("p", "l", "r")
    with pytest.raises(RuntimeError, match="no credentials"):
        app._get_access_token()
    monkeypatch.setitem(sys.modules, "google.auth", None)
    with pytest.raises(ImportError, match="google-auth"):
        app._get_access_token()


def test_parallel_tool_ids_are_scoped_to_native_invocations():
    from understudy.http import _parse_events

    events = []
    for kind in ["functionCall", "functionResponse"]:
        events.extend(
            json.dumps(
                {
                    "author": agent,
                    "invocationId": "run",
                    "nodeInfo": {"path": f"root@1/{agent}@1"},
                    "content": {
                        "parts": [
                            {
                                kind: {
                                    "id": "same",
                                    "name": "lookup",
                                    "response": {"agent": agent},
                                }
                            }
                        ]
                    },
                }
            )
            for agent in ["a", "b"]
        )
    response = _parse_events(events, None)
    assert [c.result for c in response.tool_calls] == [{"agent": "a"}, {"agent": "b"}]
    assert len({c.execution_id for c in response.tool_calls}) == 2
