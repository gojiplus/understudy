"""Unit tests for understudy core — no API keys or LLM calls needed."""

import json

import pytest
import yaml

from understudy import (
    AgentResponse,
    AgentTransfer,
    EvaluationStorage,
    Expectations,
    MockToolkit,
    Persona,
    PersonaPreset,
    RunStorage,
    Scene,
    Suite,
    ToolCall,
    ToolError,
    Trace,
    TraceStorage,
    Turn,
    check,
    evaluate,
    evaluate_batch,
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

    def test_summary(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="hi"),
                Turn(
                    role="agent",
                    content="hello",
                    tool_calls=[ToolCall(tool_name="forbidden_tool", arguments={})],
                ),
            ],
        )
        expectations = Expectations(forbidden_tools=["forbidden_tool"])
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
                expectations=Expectations(),
            )

            from understudy.check import check as check_fn

            check_result = check_fn(trace, scene.expectations)
            storage.save(trace, scene, check_result=check_result)

        summary = storage.get_summary()
        assert summary["total_runs"] == 3
        assert summary["tool_usage"]["lookup_order"] == 3


# --- Suite ---


class MockAgentApp:
    """A minimal mock implementation of AgentApp for testing."""

    def __init__(self, response_content: str = "Hi there!", terminal_state: str = "done"):
        self.response_content = response_content
        self.terminal_state = terminal_state

    def start(self, mocks=None):
        pass

    def send(self, message: str) -> AgentResponse:
        return AgentResponse(content=self.response_content, terminal_state=self.terminal_state)

    def stop(self):
        pass


class TestSuite:
    def test_run_with_tags(self, tmp_path):
        scene = Scene(
            id="tag_test_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
            expectations=Expectations(),
        )
        suite = Suite([scene])
        storage = RunStorage(path=tmp_path / "runs")

        app = MockAgentApp()
        tags = {"version": "v1", "model": "gpt-4"}
        suite.run(app, storage=storage, tags=tags)

        runs = storage.list_runs()
        assert len(runs) == 1

        run_data = storage.load(runs[0])
        assert run_data["metadata"]["tags"] == {"version": "v1", "model": "gpt-4"}


# --- Metrics ---


class TestMetrics:
    def test_trace_metrics_totals(self):
        from understudy.trace import TraceMetrics, TurnMetrics

        metrics = TraceMetrics(
            turns=[
                TurnMetrics(input_tokens=100, output_tokens=50, thinking_tokens=10, latency_ms=500),
                TurnMetrics(
                    input_tokens=200, output_tokens=100, thinking_tokens=20, latency_ms=600
                ),
            ]
        )
        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 150
        assert metrics.total_thinking_tokens == 30
        assert metrics.total_tokens == 480
        assert metrics.agent_time_ms == 1100
        assert metrics.avg_turn_latency_ms == 550.0

    def test_trace_with_metrics(self):
        from understudy.trace import TraceMetrics, TurnMetrics

        trace = Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="hello"),
                Turn(role="agent", content="hi"),
            ],
            metrics=TraceMetrics(
                turns=[
                    TurnMetrics(input_tokens=100, output_tokens=50, latency_ms=500),
                ]
            ),
        )
        assert trace.metrics.total_tokens == 150

    def test_efficiency_metric(self):
        from understudy.metrics import MetricRegistry
        from understudy.trace import TraceMetrics, TurnMetrics

        trace = Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="hello"),
                Turn(role="agent", content="hi"),
            ],
            metrics=TraceMetrics(
                turns=[
                    TurnMetrics(input_tokens=100, output_tokens=50, latency_ms=500),
                ]
            ),
        )
        expectations = Expectations()
        result = MetricRegistry.compute("efficiency", trace, expectations)
        assert result.name == "efficiency"
        assert result.value["total_tokens"] == 150
        assert result.value["turn_count"] == 2

    def test_resolution_match_metric_pass(self):
        from understudy.metrics import MetricRegistry

        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="done")],
            terminal_state="completed",
        )
        expectations = Expectations(expected_resolution="completed")
        result = MetricRegistry.compute("resolution_match", trace, expectations)
        assert result.passed is True

    def test_resolution_match_metric_fail(self):
        from understudy.metrics import MetricRegistry

        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="done")],
            terminal_state="failed",
        )
        expectations = Expectations(expected_resolution="completed")
        result = MetricRegistry.compute("resolution_match", trace, expectations)
        assert result.passed is False

    def test_tool_trajectory_metric(self):
        from understudy.metrics import MetricRegistry

        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    tool_calls=[
                        ToolCall(tool_name="lookup_order", arguments={}),
                        ToolCall(tool_name="get_return_policy", arguments={}),
                        ToolCall(tool_name="lookup_order", arguments={}),
                    ],
                )
            ],
        )
        expectations = Expectations()
        result = MetricRegistry.compute("tool_trajectory", trace, expectations)
        assert result.name == "tool_trajectory"
        assert result.value["total_calls"] == 3
        assert "lookup_order" in result.value["unique_tools"]
        assert result.value["sequence"] == ["lookup_order", "get_return_policy", "lookup_order"]

    def test_check_with_metrics(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="ok",
                    tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                )
            ],
            terminal_state="completed",
        )
        expectations = Expectations(
            required_tools=["lookup_order"],
            expected_resolution="completed",
            metrics=["efficiency", "tool_trajectory"],
        )
        result = check(trace, expectations)
        assert result.passed
        assert "efficiency" in result.metrics
        assert "tool_trajectory" in result.metrics
        assert result.metric("efficiency") is not None

    def test_check_expected_resolution_fail(self):
        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="ok")],
            terminal_state="failed",
        )
        expectations = Expectations(expected_resolution="completed")
        result = check(trace, expectations)
        assert not result.passed
        assert any(c.label == "expected_resolution" for c in result.failed_checks)


class TestStateSnapshots:
    def test_state_snapshot_in_trace(self):
        from understudy.trace import StateSnapshot

        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="ok")],
            state_snapshots=[
                StateSnapshot(turn_number=1, state={"order_id": "ORD-123"}),
                StateSnapshot(turn_number=2, state={"order_id": "ORD-123", "status": "approved"}),
            ],
        )
        assert len(trace.state_snapshots) == 2
        assert trace.state_snapshots[0].state["order_id"] == "ORD-123"
        assert trace.state_snapshots[1].state["status"] == "approved"


# --- TraceStorage ---


class TestTraceStorage:
    def test_save_and_load(self, tmp_path):
        storage = TraceStorage(path=tmp_path / "traces")

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

        trace_id = storage.save(trace, scene, sim_index=0)
        assert "test_scene_0_" in trace_id

        data = storage.load(trace_id)
        assert data["trace"].scene_id == "test_scene"
        assert data["scene"].id == "test_scene"
        assert data["metadata"]["sim_index"] == 0

    def test_list_traces(self, tmp_path):
        storage = TraceStorage(path=tmp_path / "traces")

        for i in range(3):
            trace = Trace(scene_id=f"scene_{i}", turns=[], terminal_state="done")
            scene = Scene(
                id=f"scene_{i}",
                starting_prompt="hi",
                conversation_plan="test",
                persona=Persona(description="test"),
            )
            storage.save(trace, scene, sim_index=0)

        traces = storage.list_traces()
        assert len(traces) == 3

    def test_delete(self, tmp_path):
        storage = TraceStorage(path=tmp_path / "traces")

        trace = Trace(scene_id="test", turns=[], terminal_state="done")
        scene = Scene(
            id="test",
            starting_prompt="hi",
            conversation_plan="test",
            persona=Persona(description="test"),
        )
        trace_id = storage.save(trace, scene, sim_index=0)

        storage.delete(trace_id)
        assert len(storage.list_traces()) == 0


# --- EvaluationStorage ---


class TestEvaluationStorage:
    def test_save_and_load(self, tmp_path):
        from understudy.check import CheckItem, CheckResult

        storage = EvaluationStorage(path=tmp_path / "results")

        check_result = CheckResult(
            checks=[
                CheckItem(label="required_tool", passed=True, detail="lookup_order called"),
            ]
        )

        result_id = storage.save(trace_id="test_trace", check_result=check_result)
        assert "test_trace_eval" in result_id

        data = storage.load(result_id)
        assert data["trace_id"] == "test_trace"
        assert data["passed"] is True
        assert len(data["checks"]) == 1

    def test_list_results(self, tmp_path):
        from understudy.check import CheckResult

        storage = EvaluationStorage(path=tmp_path / "results")

        for i in range(3):
            storage.save(trace_id=f"trace_{i}", check_result=CheckResult())

        results = storage.list_results()
        assert len(results) == 3


# --- Evaluate Functions ---


class TestEvaluateFunctions:
    def test_evaluate_single_trace(self):
        trace = Trace(
            scene_id="test",
            turns=[
                Turn(
                    role="agent",
                    content="done",
                    tool_calls=[ToolCall(tool_name="lookup_order", arguments={})],
                )
            ],
            terminal_state="completed",
        )
        expectations = Expectations(
            required_tools=["lookup_order"],
            expected_resolution="completed",
        )
        result = evaluate(trace, expectations)
        assert result.passed
        assert len(result.checks) == 2

    def test_evaluate_with_metrics_override(self):
        trace = Trace(
            scene_id="test",
            turns=[Turn(role="agent", content="done")],
            terminal_state="completed",
        )
        expectations = Expectations(expected_resolution="completed")
        result = evaluate(trace, expectations, metrics=["efficiency"])
        assert "efficiency" in result.metrics

    def test_evaluate_batch_from_list(self, tmp_path):
        traces = [
            Trace(
                scene_id="test_1",
                turns=[Turn(role="agent", content="done")],
                terminal_state="completed",
            ),
            Trace(
                scene_id="test_2",
                turns=[Turn(role="agent", content="fail")],
                terminal_state="failed",
            ),
        ]
        expectations = Expectations(expected_resolution="completed")

        results = evaluate_batch(traces, expectations=expectations)
        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed

    def test_evaluate_batch_from_storage(self, tmp_path):
        storage = TraceStorage(path=tmp_path / "traces")

        for i in range(2):
            trace = Trace(
                scene_id=f"scene_{i}",
                turns=[Turn(role="agent", content="ok")],
                terminal_state="completed",
            )
            scene = Scene(
                id=f"scene_{i}",
                starting_prompt="hi",
                conversation_plan="test",
                persona=Persona(description="test"),
                expectations=Expectations(expected_resolution="completed"),
            )
            storage.save(trace, scene, sim_index=0)

        output_path = tmp_path / "results"
        results = evaluate_batch(traces=tmp_path / "traces", output=output_path)
        assert len(results) == 2
        assert all(r.passed for r in results)

        result_storage = EvaluationStorage(path=output_path)
        assert len(result_storage.list_results()) == 2


# --- Suite with n_sims ---


class TestSuiteWithNSims:
    def test_run_with_n_sims(self, tmp_path):
        scene = Scene(
            id="nsims_test_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
            expectations=Expectations(),
        )
        suite = Suite([scene])
        storage = RunStorage(path=tmp_path / "runs")

        app = MockAgentApp()
        results = suite.run(app, storage=storage, n_sims=3)

        assert len(results.results) == 3
        scene_ids = [r.scene_id for r in results.results]
        assert "nsims_test_scene" in scene_ids
        assert "nsims_test_scene_1" in scene_ids
        assert "nsims_test_scene_2" in scene_ids

    def test_run_multiple_scenes_with_n_sims(self, tmp_path):
        scenes = [
            Scene(
                id=f"scene_{i}",
                starting_prompt="hello",
                conversation_plan="greet",
                persona=Persona(description="friendly"),
                expectations=Expectations(),
            )
            for i in range(2)
        ]
        suite = Suite(scenes)
        storage = RunStorage(path=tmp_path / "runs")

        app = MockAgentApp()
        results = suite.run(app, storage=storage, n_sims=2)

        assert len(results.results) == 4


# --- simulate_batch Tests ---


class TestSimulateBatch:
    def test_simulate_batch_from_list(self, tmp_path):
        from understudy import simulate_batch

        scenes = [
            Scene(
                id=f"batch_scene_{i}",
                starting_prompt="hello",
                conversation_plan="greet",
                persona=Persona(description="friendly"),
            )
            for i in range(2)
        ]

        app = MockAgentApp()
        traces = simulate_batch(app, scenes, n_sims=1, parallel=1)

        assert len(traces) == 2
        assert all(t.scene_id.startswith("batch_scene_") for t in traces)

    def test_simulate_batch_with_n_sims(self, tmp_path):
        from understudy import simulate_batch

        scene = Scene(
            id="nsims_batch_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
        )

        app = MockAgentApp()
        traces = simulate_batch(app, [scene], n_sims=3, parallel=1)

        assert len(traces) == 3
        assert all(t.scene_id == "nsims_batch_scene" for t in traces)

    def test_simulate_batch_with_output(self, tmp_path):
        from understudy import simulate_batch

        scene = Scene(
            id="output_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
        )

        output_path = tmp_path / "traces"
        app = MockAgentApp()
        traces = simulate_batch(app, [scene], n_sims=2, output=output_path)

        assert len(traces) == 2

        storage = TraceStorage(path=output_path)
        saved_traces = storage.list_traces()
        assert len(saved_traces) == 2

    def test_simulate_batch_with_tags(self, tmp_path):
        from understudy import simulate_batch

        scene = Scene(
            id="tagged_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
        )

        output_path = tmp_path / "traces"
        app = MockAgentApp()
        simulate_batch(app, [scene], n_sims=1, output=output_path, tags={"version": "v1"})

        storage = TraceStorage(path=output_path)
        trace_id = storage.list_traces()[0]
        data = storage.load(trace_id)
        assert data["metadata"]["tags"] == {"version": "v1"}

    def test_simulate_batch_from_directory(self, tmp_path):
        import yaml

        from understudy import simulate_batch

        scenes_dir = tmp_path / "scenes"
        scenes_dir.mkdir()

        for i in range(2):
            scene_data = {
                "id": f"file_scene_{i}",
                "starting_prompt": "hello",
                "conversation_plan": "greet",
                "persona": "cooperative",
            }
            (scenes_dir / f"scene_{i}.yaml").write_text(yaml.dump(scene_data))

        app = MockAgentApp()
        traces = simulate_batch(app, scenes_dir, n_sims=1, parallel=1)

        assert len(traces) == 2

    def test_simulate_batch_from_single_file(self, tmp_path):
        import yaml

        from understudy import simulate_batch

        scene_data = {
            "id": "single_file_scene",
            "starting_prompt": "hello",
            "conversation_plan": "greet",
            "persona": "cooperative",
        }
        scene_file = tmp_path / "scene.yaml"
        scene_file.write_text(yaml.dump(scene_data))

        app = MockAgentApp()
        traces = simulate_batch(app, scene_file, n_sims=1)

        assert len(traces) == 1
        assert traces[0].scene_id == "single_file_scene"

    def test_simulate_batch_parallel(self, tmp_path):
        from understudy import simulate_batch

        scenes = [
            Scene(
                id=f"parallel_scene_{i}",
                starting_prompt="hello",
                conversation_plan="greet",
                persona=Persona(description="friendly"),
            )
            for i in range(3)
        ]

        app = MockAgentApp()
        traces = simulate_batch(app, scenes, n_sims=1, parallel=2)

        assert len(traces) == 3

    def test_simulate_batch_with_mocks(self, tmp_path):
        from understudy import MockToolkit, simulate_batch

        scene = Scene(
            id="mock_scene",
            starting_prompt="hello",
            conversation_plan="greet",
            persona=Persona(description="friendly"),
        )

        mocks = MockToolkit()

        @mocks.handle("test_tool")
        def test_tool():
            return "mocked result"

        app = MockAgentApp()
        traces = simulate_batch(app, [scene], n_sims=1, mocks=mocks)

        assert len(traces) == 1


# --- evaluate_batch comprehensive tests ---


class TestEvaluateBatchComprehensive:
    def test_evaluate_batch_parallel(self, tmp_path):
        traces = [
            Trace(
                scene_id=f"parallel_eval_{i}",
                turns=[Turn(role="agent", content="done")],
                terminal_state="completed",
            )
            for i in range(4)
        ]
        expectations = Expectations(expected_resolution="completed")

        results = evaluate_batch(traces, expectations=expectations, parallel=2)

        assert len(results) == 4
        assert all(r.passed for r in results)

    def test_evaluate_batch_with_output(self, tmp_path):
        traces = [
            Trace(
                scene_id="output_eval",
                turns=[Turn(role="agent", content="done")],
                terminal_state="completed",
            )
        ]
        expectations = Expectations(expected_resolution="completed")
        output_path = tmp_path / "results"

        results = evaluate_batch(traces, expectations=expectations, output=output_path)

        assert len(results) == 1
        result_storage = EvaluationStorage(path=output_path)
        assert len(result_storage.list_results()) == 1

    def test_evaluate_batch_with_metrics(self, tmp_path):
        from understudy.trace import TraceMetrics, TurnMetrics

        traces = [
            Trace(
                scene_id="metrics_eval",
                turns=[Turn(role="agent", content="done")],
                terminal_state="completed",
                metrics=TraceMetrics(
                    turns=[TurnMetrics(input_tokens=100, output_tokens=50, latency_ms=500)]
                ),
            )
        ]
        expectations = Expectations()

        results = evaluate_batch(traces, expectations=expectations, metrics=["efficiency"])

        assert len(results) == 1
        assert "efficiency" in results[0].check_result.metrics

    def test_evaluate_batch_handles_exceptions(self, tmp_path):
        class BadTrace:
            scene_id = "bad"

        traces = [BadTrace()]  # type: ignore
        expectations = Expectations()

        results = evaluate_batch(traces, expectations=expectations)  # type: ignore

        assert len(results) == 1
        assert results[0].error is not None
        assert "Traceback" in results[0].error

    def test_evaluate_batch_mixed_results(self, tmp_path):
        traces = [
            Trace(
                scene_id="pass",
                turns=[Turn(role="agent", content="done")],
                terminal_state="completed",
            ),
            Trace(
                scene_id="fail",
                turns=[Turn(role="agent", content="fail")],
                terminal_state="failed",
            ),
        ]
        expectations = Expectations(expected_resolution="completed")

        results = evaluate_batch(traces, expectations=expectations)

        assert len(results) == 2
        passed = sum(1 for r in results if r.passed)
        assert passed == 1

    def test_evaluate_batch_without_expectations(self, tmp_path):
        traces = [
            Trace(
                scene_id="no_exp",
                turns=[Turn(role="agent", content="done")],
            )
        ]

        results = evaluate_batch(traces)

        assert len(results) == 1
        assert results[0].passed
