"""Unit tests for understudy core — no API keys or LLM calls needed."""

import json

import pytest
import yaml

from understudy import (
    AgentTransfer,
    Expectations,
    MockToolkit,
    Persona,
    PersonaPreset,
    RunStorage,
    Scene,
    ToolCall,
    ToolError,
    Trace,
    Turn,
    check,
)

# --- Persona ---


class TestPersona:
    def test_from_preset_string(self):
        p = Persona.from_preset("cooperative")
        assert "direct" in p.description.lower() or "helpful" in p.description.lower()

    def test_from_preset_enum(self):
        p = Persona.from_preset(PersonaPreset.ADVERSARIAL)
        assert len(p.behaviors) > 0

    def test_class_attributes(self):
        assert Persona.COOPERATIVE.description
        assert Persona.ADVERSARIAL.description
        assert Persona.FRUSTRATED_BUT_COOPERATIVE.description

    def test_to_prompt(self):
        p = Persona(description="test persona", behaviors=["behavior 1"])
        prompt = p.to_prompt()
        assert "test persona" in prompt
        assert "behavior 1" in prompt

    def test_custom_persona(self):
        p = Persona(
            description="elderly, not tech-savvy",
            behaviors=["asks for clarification"],
        )
        assert p.description == "elderly, not tech-savvy"


# --- Scene ---


class TestScene:
    def test_from_yaml(self, tmp_path):
        scene_data = {
            "id": "test_scene",
            "starting_prompt": "Hello",
            "conversation_plan": "Ask about order",
            "persona": "cooperative",
            "max_turns": 10,
            "context": {"orders": {"ORD-1": {"status": "delivered"}}},
            "expectations": {
                "required_tools": ["lookup_order"],
                "forbidden_tools": ["issue_refund"],
                "allowed_terminal_states": ["resolved"],
            },
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(scene_data))

        scene = Scene.from_file(path)
        assert scene.id == "test_scene"
        assert scene.persona.description  # resolved from preset
        assert scene.expectations.required_tools == ["lookup_order"]
        assert scene.max_turns == 10

    def test_from_json(self, tmp_path):
        scene_data = {
            "id": "json_scene",
            "starting_prompt": "Hi",
            "conversation_plan": "Ask about order",
            "persona": {"description": "custom", "behaviors": ["be nice"]},
            "expectations": {},
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scene_data))

        scene = Scene.from_file(path)
        assert scene.id == "json_scene"
        assert scene.persona.description == "custom"

    def test_custom_persona_in_yaml(self, tmp_path):
        scene_data = {
            "id": "custom_persona_scene",
            "starting_prompt": "Hello",
            "conversation_plan": "Be vague",
            "persona": {
                "description": "aggressive customer",
                "behaviors": ["demands refund", "threatens legal action"],
            },
            "expectations": {},
        }
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump(scene_data))

        scene = Scene.from_file(path)
        assert "aggressive" in scene.persona.description
        assert len(scene.persona.behaviors) == 2


# --- Trace ---


class TestTrace:
    def _make_trace(self) -> Trace:
        return Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="I want to return something"),
                Turn(
                    role="agent",
                    content="Let me look that up",
                    tool_calls=[
                        ToolCall(
                            tool_name="lookup_order",
                            arguments={"order_id": "ORD-10027"},
                            result={"status": "delivered"},
                        ),
                        ToolCall(
                            tool_name="get_return_policy",
                            arguments={"category": "personal_audio"},
                            result={"returnable": False},
                        ),
                    ],
                ),
                Turn(role="user", content="But it's defective!"),
                Turn(
                    role="agent",
                    content="I understand, but policy prevents this return.",
                    tool_calls=[],
                ),
            ],
            terminal_state="return_denied_policy",
        )

    def test_called(self):
        trace = self._make_trace()
        assert trace.called("lookup_order")
        assert trace.called("get_return_policy")
        assert not trace.called("create_return")
        assert not trace.called("issue_refund")

    def test_called_with_args(self):
        trace = self._make_trace()
        assert trace.called("lookup_order", order_id="ORD-10027")
        assert not trace.called("lookup_order", order_id="ORD-99999")

    def test_calls_to(self):
        trace = self._make_trace()
        calls = trace.calls_to("lookup_order")
        assert len(calls) == 1
        assert calls[0].arguments["order_id"] == "ORD-10027"

    def test_call_sequence(self):
        trace = self._make_trace()
        assert trace.call_sequence() == ["lookup_order", "get_return_policy"]

    def test_tool_calls_flat(self):
        trace = self._make_trace()
        assert len(trace.tool_calls) == 2

    def test_turn_count(self):
        trace = self._make_trace()
        assert trace.turn_count == 4

    def test_terminal_state(self):
        trace = self._make_trace()
        assert trace.terminal_state == "return_denied_policy"

    def test_conversation_text(self):
        trace = self._make_trace()
        text = trace.conversation_text()
        assert "[USER]:" in text
        assert "[AGENT]:" in text
        assert "lookup_order" in text


# --- Check ---


class TestCheck:
    def test_all_pass(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="denied",
                    tool_calls=[
                        ToolCall(tool_name="lookup_order", arguments={}),
                        ToolCall(tool_name="get_return_policy", arguments={}),
                    ],
                )
            ],
            terminal_state="return_denied_policy",
        )
        expectations = Expectations(
            required_tools=["lookup_order", "get_return_policy"],
            forbidden_tools=["create_return"],
            allowed_terminal_states=["return_denied_policy"],
        )
        result = check(trace, expectations)
        assert result.passed

    def test_missing_required_tool(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                )
            ],
            terminal_state="return_denied_policy",
        )
        expectations = Expectations(
            required_tools=["lookup_order", "get_return_policy"],
        )
        result = check(trace, expectations)
        assert not result.passed
        assert len(result.failed_checks) == 1

    def test_forbidden_tool_called(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    tool_calls=[ToolCall(tool_name="create_return", arguments={})],
                )
            ],
            terminal_state="return_created",
        )
        expectations = Expectations(
            forbidden_tools=["create_return"],
        )
        result = check(trace, expectations)
        assert not result.passed

    def test_wrong_terminal_state(self):
        trace = Trace(
            scene_id="test",
            turns=[],
            terminal_state="return_created",
        )
        expectations = Expectations(
            allowed_terminal_states=["return_denied_policy"],
        )
        result = check(trace, expectations)
        assert not result.passed

    def test_summary(self):
        trace = Trace(scene_id="test", turns=[], terminal_state="bad_state")
        expectations = Expectations(allowed_terminal_states=["good_state"])
        result = check(trace, expectations)
        summary = result.summary()
        assert "✗" in summary


# --- MockToolkit ---


class TestMockToolkit:
    def test_handle_decorator(self):
        mocks = MockToolkit()

        @mocks.handle("my_tool")
        def my_tool(x: int) -> int:
            return x * 2

        assert mocks.call("my_tool", x=5) == 10

    def test_multiple_handlers(self):
        mocks = MockToolkit()

        @mocks.handle("lookup_order")
        def lookup_order(order_id: str) -> dict:
            return {"order_id": order_id, "status": "delivered"}

        @mocks.handle("create_return")
        def create_return(order_id: str, item_sku: str, reason: str) -> dict:
            return {"return_id": "RET-001", "order_id": order_id}

        result = mocks.call("lookup_order", order_id="ORD-1")
        assert result["status"] == "delivered"

        result2 = mocks.call(
            "create_return", order_id="ORD-1", item_sku="SKU-1", reason="too small"
        )
        assert result2["return_id"] == "RET-001"

    def test_handler_raises_tool_error(self):
        mocks = MockToolkit()

        @mocks.handle("lookup_order")
        def lookup_order(order_id: str) -> dict:
            raise ToolError(f"Order {order_id} not found")

        with pytest.raises(ToolError):
            mocks.call("lookup_order", order_id="ORD-MISSING")

    def test_available_tools(self):
        mocks = MockToolkit()

        @mocks.handle("tool_a")
        def tool_a():
            pass

        @mocks.handle("tool_b")
        def tool_b():
            pass

        tools = mocks.available_tools
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_get_handler(self):
        mocks = MockToolkit()

        @mocks.handle("my_tool")
        def my_tool():
            return "result"

        assert mocks.get_handler("my_tool") is not None
        assert mocks.get_handler("unknown_tool") is None

    def test_call_unregistered_tool_raises(self):
        mocks = MockToolkit()
        with pytest.raises(KeyError):
            mocks.call("unknown_tool")


# --- Subagent Support ---


class TestSubagentSupport:
    def _make_trace_with_agents(self) -> Trace:
        return Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="I need help with billing"),
                Turn(
                    role="agent",
                    content="Let me transfer you",
                    agent_name="customer_service",
                    tool_calls=[
                        ToolCall(
                            tool_name="lookup_customer",
                            arguments={"email": "test@example.com"},
                            agent_name="customer_service",
                        ),
                    ],
                ),
                Turn(
                    role="agent",
                    content="I can help with billing",
                    agent_name="billing_agent",
                    tool_calls=[
                        ToolCall(
                            tool_name="get_invoice",
                            arguments={"id": "INV-001"},
                            agent_name="billing_agent",
                        ),
                        ToolCall(
                            tool_name="process_payment",
                            arguments={"amount": 100},
                            agent_name="billing_agent",
                        ),
                    ],
                ),
            ],
            agent_transfers=[
                AgentTransfer(from_agent="customer_service", to_agent="billing_agent"),
            ],
            terminal_state="payment_processed",
        )

    def test_agents_invoked(self):
        trace = self._make_trace_with_agents()
        agents = trace.agents_invoked()
        assert "customer_service" in agents
        assert "billing_agent" in agents
        assert len(agents) == 2

    def test_agent_called(self):
        trace = self._make_trace_with_agents()
        assert trace.agent_called("billing_agent", "get_invoice")
        assert trace.agent_called("billing_agent", "process_payment")
        assert trace.agent_called("customer_service", "lookup_customer")
        assert not trace.agent_called("customer_service", "get_invoice")

    def test_calls_by_agent(self):
        trace = self._make_trace_with_agents()
        billing_calls = trace.calls_by_agent("billing_agent")
        assert len(billing_calls) == 2
        cs_calls = trace.calls_by_agent("customer_service")
        assert len(cs_calls) == 1

    def test_agent_transfers(self):
        trace = self._make_trace_with_agents()
        assert len(trace.agent_transfers) == 1
        assert trace.agent_transfers[0].from_agent == "customer_service"
        assert trace.agent_transfers[0].to_agent == "billing_agent"

    def test_conversation_text_with_agent_names(self):
        trace = self._make_trace_with_agents()
        text = trace.conversation_text()
        assert "[CUSTOMER_SERVICE]:" in text
        assert "[BILLING_AGENT]:" in text


class TestAgentExpectations:
    def test_required_agents_pass(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    agent_name="billing_agent",
                    tool_calls=[
                        ToolCall(
                            tool_name="process_payment", arguments={}, agent_name="billing_agent"
                        )
                    ],
                ),
            ],
            terminal_state="done",
        )
        expectations = Expectations(required_agents=["billing_agent"])
        result = check(trace, expectations)
        assert result.passed

    def test_required_agents_fail(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    agent_name="customer_service",
                    tool_calls=[],
                ),
            ],
            terminal_state="done",
        )
        expectations = Expectations(required_agents=["billing_agent"])
        result = check(trace, expectations)
        assert not result.passed

    def test_forbidden_agents_pass(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(role="agent", content="ok", agent_name="customer_service", tool_calls=[]),
            ],
            terminal_state="done",
        )
        expectations = Expectations(forbidden_agents=["admin_agent"])
        result = check(trace, expectations)
        assert result.passed

    def test_forbidden_agents_fail(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(role="agent", content="ok", agent_name="admin_agent", tool_calls=[]),
            ],
            terminal_state="done",
        )
        expectations = Expectations(forbidden_agents=["admin_agent"])
        result = check(trace, expectations)
        assert not result.passed

    def test_required_agent_tools(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    agent_name="billing_agent",
                    tool_calls=[
                        ToolCall(
                            tool_name="process_payment", arguments={}, agent_name="billing_agent"
                        ),
                    ],
                ),
            ],
            terminal_state="done",
        )
        expectations = Expectations(required_agent_tools={"billing_agent": ["process_payment"]})
        result = check(trace, expectations)
        assert result.passed


# --- Storage ---


class TestRunStorage:
    def test_save_and_load(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace = Trace(
            scene_id="test_scene",
            turns=[Turn(role="user", content="hello")],
            terminal_state="done",
        )
        scene = Scene(
            id="test_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
        )

        run_id = storage.save(trace, scene)
        assert run_id.startswith("test_scene_")

        data = storage.load(run_id)
        assert data["trace"].scene_id == "test_scene"
        assert data["scene"].id == "test_scene"
        assert data["metadata"]["terminal_state"] == "done"

    def test_list_runs(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for i in range(3):
            trace = Trace(scene_id=f"scene_{i}", turns=[], terminal_state="done")
            scene = Scene(
                id=f"scene_{i}",
                starting_prompt="hi",
                conversation_plan="test",
                persona=Persona(description="test"),
            )
            storage.save(trace, scene)

        runs = storage.list_runs()
        assert len(runs) == 3

    def test_delete(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        trace = Trace(scene_id="test", turns=[], terminal_state="done")
        scene = Scene(
            id="test",
            starting_prompt="hi",
            conversation_plan="test",
            persona=Persona(description="test"),
        )
        run_id = storage.save(trace, scene)

        storage.delete(run_id)
        assert len(storage.list_runs()) == 0

    def test_get_summary(self, tmp_path):
        storage = RunStorage(path=tmp_path / "runs")

        for i in range(3):
            trace = Trace(
                scene_id=f"scene_{i}",
                turns=[
                    Turn(
                        role="agent",
                        content="ok",
                        tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                    )
                ],
                terminal_state="done" if i < 2 else "failed",
            )
            scene = Scene(
                id=f"scene_{i}",
                starting_prompt="hi",
                conversation_plan="test",
                persona=Persona(description="test"),
                expectations=Expectations(allowed_terminal_states=["done"]),
            )

            from understudy.check import check as check_fn

            check_result = check_fn(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result)

        summary = storage.get_summary()
        assert summary["total_runs"] == 3
        assert summary["tool_usage"]["lookup_order"] == 3
