"""Tests for the HTTP simulator API server."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from understudy.server import app, session_manager
from understudy.server.models import Affordance
from understudy.server.ui_simulator import UISimulator


class MockBackend:
    """Mock LLM backend that returns predictable actions."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return '{"done": true, "reason": "finished"}'


@pytest.fixture
def client():
    return TestClient(app)


class TestCreateSession:
    def test_create_session_success(self, client):
        response = client.post(
            "/sessions",
            json={
                "scene": {
                    "id": "test_scene",
                    "starting_prompt": "I need help",
                    "conversation_plan": "Ask for help politely",
                    "persona": "cooperative",
                },
                "simulatorModel": "gpt-4o",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "sessionId" in data
        assert "firstAction" in data
        assert data["firstAction"]["type"] == "type"
        assert data["firstAction"]["value"] == "I need help"

    def test_create_session_with_expectations(self, client):
        response = client.post(
            "/sessions",
            json={
                "scene": {
                    "id": "test_scene",
                    "starting_prompt": "Return my order",
                    "conversation_plan": "Request return for order 12345",
                    "persona": "frustrated_but_cooperative",
                    "expectations": {
                        "required_tools": ["create_return"],
                        "forbidden_tools": ["delete_account"],
                    },
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["firstAction"]["value"] == "Return my order"

    def test_create_session_custom_persona(self, client):
        response = client.post(
            "/sessions",
            json={
                "scene": {
                    "id": "custom_scene",
                    "starting_prompt": "Hello",
                    "conversation_plan": "Be friendly",
                    "persona": {
                        "description": "A very friendly user",
                        "behaviors": ["Always says please", "Very patient"],
                    },
                },
            },
        )
        assert response.status_code == 200


class TestProcessTurn:
    def test_turn_session_not_found(self, client):
        response = client.post(
            "/sessions/nonexistent-id/turn",
            json={
                "displayedContent": "Hello!",
                "affordances": [],
            },
        )
        assert response.status_code == 404

    def test_turn_with_mocked_backend(self, client):
        def create_backend(_model: str):
            return MockBackend(
                [
                    '{"type": "type", "target": {"id": "input", "selector": "input.chat"}, '
                    '"value": "My order number is 12345"}',
                    '{"done": true, "reason": "finished"}',
                ]
            )

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "test_turn",
                        "starting_prompt": "Help me",
                        "conversation_plan": "Provide order number when asked",
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            turn_resp = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "What is your order number?",
                    "affordances": [
                        {"id": "input", "type": "text_input", "selector": "input.chat"},
                        {"id": "send", "type": "button", "selector": "button.send"},
                    ],
                },
            )
            assert turn_resp.status_code == 200
            data = turn_resp.json()
            assert data["status"] == "continue"
            assert data["action"]["type"] == "type"
            assert "12345" in data["action"]["value"]

            turn_resp2 = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Thank you, your return is processed.",
                    "affordances": [
                        {"id": "input", "type": "text_input", "selector": "input.chat"},
                    ],
                },
            )
            assert turn_resp2.json()["status"] == "done"

        finally:
            session_manager.set_backend_factory(None)

    def test_turn_max_turns_reached(self, client):
        def create_backend(_model: str):
            return MockBackend(
                [
                    '{"type": "type", "target": {"id": "input", "selector": "input"}, '
                    '"value": "message"}'
                    for _ in range(100)
                ]
            )

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "max_turns_test",
                        "starting_prompt": "Start",
                        "conversation_plan": "Keep talking",
                        "max_turns": 2,
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            for i in range(3):
                turn_resp = client.post(
                    f"/sessions/{session_id}/turn",
                    json={
                        "displayedContent": f"Response {i}",
                        "affordances": [
                            {"id": "input", "type": "text_input", "selector": "input"},
                        ],
                    },
                )
                data = turn_resp.json()
                if data["status"] == "done":
                    assert data["reason"] == "max_turns_reached"
                    break

        finally:
            session_manager.set_backend_factory(None)

    def test_turn_with_tool_calls(self, client):
        def create_backend(_model: str):
            return MockBackend(['{"done": true, "reason": "finished"}'])

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "tool_test",
                        "starting_prompt": "Create return",
                        "conversation_plan": "Request return",
                        "expectations": {"required_tools": ["create_return"]},
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            turn_resp = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Return created!",
                    "affordances": [],
                    "toolCalls": [
                        {
                            "tool_name": "create_return",
                            "arguments": {"order_id": "12345"},
                            "result": {"return_id": "RET-001"},
                        }
                    ],
                },
            )
            assert turn_resp.status_code == 200

            eval_resp = client.post(f"/sessions/{session_id}/evaluate")
            assert eval_resp.json()["passed"] is True

        finally:
            session_manager.set_backend_factory(None)


class TestEvaluate:
    def test_evaluate_session_not_found(self, client):
        response = client.post("/sessions/nonexistent/evaluate")
        assert response.status_code == 404

    def test_evaluate_with_expectations(self, client):
        def create_backend(_model: str):
            return MockBackend(['{"done": true, "reason": "finished"}'])

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "eval_test",
                        "starting_prompt": "Help",
                        "conversation_plan": "Test",
                        "expectations": {
                            "required_tools": ["tool_a", "tool_b"],
                            "forbidden_tools": ["tool_c"],
                        },
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Done",
                    "affordances": [],
                    "toolCalls": [
                        {"tool_name": "tool_a", "arguments": {}},
                    ],
                },
            )

            eval_resp = client.post(f"/sessions/{session_id}/evaluate")
            data = eval_resp.json()
            assert data["passed"] is False
            assert "2/3 checks passed" in data["summary"]

            assert any("required_tool" in c["label"] for c in data["checks"])

        finally:
            session_manager.set_backend_factory(None)


class TestGetTrace:
    def test_trace_session_not_found(self, client):
        response = client.get("/sessions/nonexistent/trace")
        assert response.status_code == 404

    def test_trace_success(self, client):
        def create_backend(_model: str):
            return MockBackend(['{"done": true, "reason": "finished"}'])

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "trace_test",
                        "starting_prompt": "Hello",
                        "conversation_plan": "Greet",
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Hi there!",
                    "affordances": [],
                },
            )

            trace_resp = client.get(f"/sessions/{session_id}/trace")
            assert trace_resp.status_code == 200
            data = trace_resp.json()
            assert data["scene_id"] == "trace_test"
            assert len(data["turns"]) >= 1

        finally:
            session_manager.set_backend_factory(None)


class TestDeleteSession:
    def test_delete_session_not_found(self, client):
        response = client.delete("/sessions/nonexistent")
        assert response.status_code == 404

    def test_delete_session_success(self, client):
        create_resp = client.post(
            "/sessions",
            json={
                "scene": {
                    "id": "delete_test",
                    "starting_prompt": "Test",
                    "conversation_plan": "Test",
                },
            },
        )
        session_id = create_resp.json()["sessionId"]

        delete_resp = client.delete(f"/sessions/{session_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["status"] == "deleted"

        get_resp = client.get(f"/sessions/{session_id}/trace")
        assert get_resp.status_code == 404


class TestUISimulator:
    def test_get_first_action(self):
        backend = MockBackend()
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Test plan",
            persona_prompt="Test persona",
        )

        affordances = [
            Affordance(id="main-input", type="text_input", selector="input.main"),
            Affordance(id="submit", type="button", selector="button.submit", label="Send"),
        ]

        action = simulator.get_first_action("Hello there", affordances)
        assert action.type == "type"
        assert action.target.id == "main-input"
        assert action.target.selector == "input.main"
        assert action.value == "Hello there"

    def test_next_action_type(self):
        backend = MockBackend(
            [
                '{"type": "type", "target": {"id": "input", "selector": "input.chat"}, '
                '"value": "test message"}'
            ]
        )
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Reply to messages",
            persona_prompt="Friendly user",
        )

        affordances = [
            Affordance(id="input", type="text_input", selector="input.chat"),
        ]

        action = simulator.next_action("Agent says hello", affordances)
        assert action.type == "type"
        assert action.value == "test message"

    def test_next_action_click(self):
        backend = MockBackend(
            ['{"type": "click", "target": {"id": "btn", "selector": "button.confirm"}}']
        )
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Click buttons",
            persona_prompt="Test user",
        )

        action = simulator.next_action("Click to confirm", [])
        assert action.type == "click"
        assert action.target.selector == "button.confirm"

    def test_next_action_done(self):
        backend = MockBackend(['{"done": true, "reason": "task completed"}'])
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Finish when done",
            persona_prompt="Test user",
        )

        action = simulator.next_action("Thank you, goodbye!", [])
        assert action is None

    def test_next_action_invalid_json(self):
        backend = MockBackend(["not valid json"])
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Test",
            persona_prompt="Test",
        )

        action = simulator.next_action("Hello", [])
        assert action.type == "wait"
        assert action.duration == 500

    def test_history_tracking(self):
        backend = MockBackend(
            [
                '{"type": "type", "target": {"id": "in", "selector": "input"}, "value": "first"}',
                '{"type": "type", "target": {"id": "in", "selector": "input"}, "value": "second"}',
            ]
        )
        simulator = UISimulator(
            backend=backend,
            conversation_plan="Track history",
            persona_prompt="Test",
        )

        simulator.next_action("Agent message 1", [])
        simulator.next_action("Agent message 2", [])

        assert len(simulator.history) == 4
        assert simulator.history[0]["role"] == "assistant"
        assert simulator.history[0]["content"] == "Agent message 1"
        assert simulator.history[1]["role"] == "user"
        assert simulator.history[1]["content"] == "first"


class TestIntegration:
    """Full integration tests simulating a complete conversation flow."""

    def test_full_return_flow(self, client):
        responses = [
            '{"type": "type", "target": {"id": "input", "selector": "input"}, '
            '"value": "Yes, order ORD-12345"}',
            '{"type": "type", "target": {"id": "input", "selector": "input"}, '
            '"value": "The item was defective"}',
            '{"done": true, "reason": "return processed"}',
        ]

        def create_backend(_model: str):
            return MockBackend(responses)

        session_manager.set_backend_factory(create_backend)

        try:
            create_resp = client.post(
                "/sessions",
                json={
                    "scene": {
                        "id": "return_flow",
                        "starting_prompt": "I want to return an item",
                        "conversation_plan": "Return item from order ORD-12345, reason: defective",
                        "persona": "cooperative",
                        "expectations": {"required_tools": ["create_return"]},
                    },
                },
            )
            session_id = create_resp.json()["sessionId"]

            affordances = [
                {"id": "input", "type": "text_input", "selector": "input"},
                {"id": "send", "type": "button", "selector": "button.send", "label": "Send"},
            ]

            turn1 = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "What is your order number?",
                    "affordances": affordances,
                },
            )
            assert turn1.json()["status"] == "continue"
            assert "ORD-12345" in turn1.json()["action"]["value"]

            turn2 = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Why do you want to return it?",
                    "affordances": affordances,
                },
            )
            assert turn2.json()["status"] == "continue"
            assert "defective" in turn2.json()["action"]["value"]

            turn3 = client.post(
                f"/sessions/{session_id}/turn",
                json={
                    "displayedContent": "Your return has been created.",
                    "affordances": affordances,
                    "toolCalls": [
                        {"tool_name": "create_return", "arguments": {"order_id": "ORD-12345"}}
                    ],
                },
            )
            assert turn3.json()["status"] == "done"

            eval_resp = client.post(f"/sessions/{session_id}/evaluate")
            assert eval_resp.json()["passed"] is True

            trace_resp = client.get(f"/sessions/{session_id}/trace")
            trace = trace_resp.json()
            assert trace["scene_id"] == "return_flow"
            assert len(trace["turns"]) >= 3

            client.delete(f"/sessions/{session_id}")

        finally:
            session_manager.set_backend_factory(None)
