"""LangGraph adapter: wraps LangGraph agents for use with understudy."""

import asyncio
import base64
import logging
import uuid
from copy import deepcopy
from importlib.metadata import version
from typing import Any

from ..execution import (
    AdapterCapabilities,
    Checkpoint,
    StateFixture,
    UnsupportedCapabilityError,
)
from ..mocks import MockToolkit
from ..runner import AgentApp, AgentResponse
from ..trace import ToolCall
from .tools import MockableToolContext, get_mock_toolkit

logger = logging.getLogger(__name__)


class LangGraphApp(AgentApp):
    """Wraps a LangGraph CompiledGraph for use with understudy.

    Usage:
        from langgraph.graph import StateGraph
        from understudy.langgraph import LangGraphApp

        # Build your LangGraph agent
        graph = StateGraph(...)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        compiled = graph.compile()

        # Test with understudy
        app = LangGraphApp(graph=compiled)
        trace = run(app, scene)
        assert trace.called("search_orders")
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        config: dict[str, Any] | None = None,
        app_id: str = "langgraph",
        tool_interception: bool = False,
    ):
        """Wrap a compiled LangGraph graph.

        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: The key used for message input in the graph state.
            config: Optional config dict passed to graph.stream().
            app_id: Application revision identifier for checkpoint validation.
            tool_interception: Declare that tools use interception wrappers
                or middleware.
        """
        self.graph = graph
        self.app_id = app_id
        self.capabilities = AdapterCapabilities(
            tools=True,
            agents=True,
            nested=True,
            state=bool(getattr(graph, "checkpointer", None)),
            checkpoints=bool(getattr(graph, "checkpointer", None)),
            interception=tool_interception,
        )
        self.input_key = input_key
        self.config = config or {}
        self.thread_id: str | None = None
        self._loop: Any = None
        self._mocks: MockToolkit | None = None

    def start(self, mocks: MockToolkit | None = None) -> None:
        """Initialize the LangGraph session."""
        try:
            from langchain_core.messages import HumanMessage  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "langchain-core package required. "
                "Install with: pip install understudy[langgraph]"
            ) from e

        if mocks is not None and not self.capabilities.interception:
            raise UnsupportedCapabilityError(
                "Declare tool_interception=True after instrumenting the tools"
            )
        self._loop = asyncio.Runner()
        self.thread_id = str(uuid.uuid4())
        self._mocks = mocks
        logger.debug("Starting LangGraph session %s", self.thread_id)

    def send(self, message: str) -> AgentResponse:
        """Run a turn on the session's loop, including asynchronous tools."""
        if self._loop is None:
            raise RuntimeError("LangGraphApp not started. Call start() first.")
        return self._loop.run(self._send_async(message))

    async def _send_async(self, message: str) -> AgentResponse:
        """Collect native streaming events from synchronous or async graph nodes."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        except ImportError as e:
            raise ImportError(
                "langchain-core package required. "
                "Install with: pip install understudy[langgraph]"
            ) from e

        if self.thread_id is None:
            raise RuntimeError("LangGraphApp not started. Call start() first.")

        logger.debug("Sending message: %s", message[:100])

        input_state = {self.input_key: [HumanMessage(content=message)]}

        config = {
            **self.config,
            "configurable": {
                **self.config.get("configurable", {}),
                "thread_id": self.thread_id,
            },
        }

        tool_calls: list[ToolCall] = []
        calls_by_namespace: dict[tuple[str, ...], list[ToolCall]] = {}
        tool_call_map: dict[tuple[tuple[str, ...], str], ToolCall] = {}
        final_content = ""
        terminal_state: str | None = None
        input_tokens = 0
        output_tokens = 0
        thinking_tokens = 0

        from .instrumentation import GraphRecorder

        recorder = GraphRecorder()
        seen_messages = set()
        if self.capabilities.state:
            try:
                previous = self.graph.get_state(config)
                if isinstance(previous.values, dict):
                    seen_messages.update(
                        m.id
                        for m in previous.values.get(self.input_key, [])
                        if getattr(m, "id", None)
                    )
            except (ValueError, AttributeError):
                pass
        config["callbacks"] = [*(config.get("callbacks") or []), recorder]
        with MockableToolContext(self._mocks or get_mock_toolkit()):
            async for namespace, chunk in self.graph.astream(
                input_state, config=config, stream_mode="updates", subgraphs=True
            ):
                for node, updates in chunk.items():
                    if node == "__interrupt__":
                        terminal_state = "interrupted"
                        continue
                    if not isinstance(updates, dict):
                        continue
                    messages = updates.get("messages", [])
                    if not isinstance(messages, list):
                        messages = [messages]

                    for msg in messages:
                        if msg.id and msg.id in seen_messages:
                            continue
                        if msg.id:
                            seen_messages.add(msg.id)
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    call = ToolCall(
                                        tool_name=tc["name"],
                                        arguments=tc.get("args", {}),
                                        call_id=tc.get("id"),
                                        agent_name=node,
                                        agent_path=(
                                            *tuple(n.split(":")[0] for n in namespace),
                                            node,
                                        ),
                                    )
                                    scopes = [
                                        e
                                        for e in recorder.executions.values()
                                        if e.agent_name == node
                                        and all(
                                            n.split(":")[0] in e.agent_path
                                            for n in namespace
                                        )
                                    ]
                                    if scopes:
                                        call.execution_id = scopes[-1].id
                                        call.agent_path = scopes[-1].agent_path
                                    tool_calls.append(call)
                                    calls_by_namespace.setdefault(namespace, []).append(
                                        call
                                    )
                                    # id is optional in LangChain's ToolCall; the
                                    # map exists only to pair a call with its later
                                    # result, so an unidentified call simply has no
                                    # entry rather than one keyed on None.
                                    call_id = tc.get("id")
                                    if call_id is not None:
                                        tool_call_map[(namespace, call_id)] = call
                                    logger.debug("Tool call: %s", tc["name"])
                            if msg.content and isinstance(msg.content, str):
                                final_content = msg.content

                            # capture token usage from response_metadata
                            if msg.usage_metadata:
                                input_tokens += msg.usage_metadata["input_tokens"]
                                output_tokens += msg.usage_metadata["output_tokens"]
                                thinking_tokens += msg.usage_metadata.get(
                                    "output_token_details", {}
                                ).get("reasoning", 0)
                            elif msg.response_metadata:
                                usage = msg.response_metadata.get("token_usage", {})
                                input_tokens += usage.get("prompt_tokens", 0) or 0
                                output_tokens += usage.get("completion_tokens", 0) or 0
                                thinking_tokens += usage.get("reasoning_tokens", 0) or 0

                        elif isinstance(msg, ToolMessage):
                            tool_call_id = msg.tool_call_id
                            key = (namespace, tool_call_id)
                            if tool_call_id and key in tool_call_map:
                                tool_call_map[key].result = msg.content
                            else:
                                tool_name = getattr(msg, "name", None)
                                if tool_name:
                                    for call in reversed(
                                        calls_by_namespace.get(namespace, [])
                                    ):
                                        if (
                                            call.tool_name == tool_name
                                            and call.result is None
                                        ):
                                            call.result = msg.content
                                            break

        # capture state snapshot from graph and detect END node
        state_snapshot = None
        try:
            graph_state = self.graph.get_state(config)
            if graph_state and hasattr(graph_state, "values"):
                state_values = graph_state.values
                if isinstance(state_values, dict):
                    terminal_state = state_values.get("terminal_state", terminal_state)
                    state_snapshot = {
                        k: v for k, v in state_values.items() if k != "messages"
                    }

        except Exception as e:
            logger.debug("Failed to capture graph state: %s", e)

        executions = list(recorder.executions.values())
        return AgentResponse(
            content=final_content,
            executions=executions,
            tool_calls=tool_calls,
            terminal_state=terminal_state,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            state_snapshot=state_snapshot,
        )

    def stop(self) -> None:
        """Clean up the LangGraph session."""
        self.thread_id = None
        self._mocks = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None
        logger.debug("LangGraph session stopped")

    def _config(self) -> dict[str, Any]:
        if self.thread_id is None:
            raise RuntimeError("LangGraphApp not started")
        return {
            **self.config,
            "configurable": {
                **self.config.get("configurable", {}),
                "thread_id": self.thread_id,
            },
        }

    def get_state(self) -> dict[str, Any]:
        """Read graph values without exposing mutable checkpoint storage."""
        return deepcopy(dict(self.graph.get_state(self._config()).values))

    def validate_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Reject checkpoints with incompatible runtime or application identity."""
        if (checkpoint.adapter, checkpoint.framework_version, checkpoint.app_id) != (
            "langgraph",
            version("langgraph"),
            self.app_id,
        ):
            raise ValueError(
                "Checkpoint adapter, framework version, or app_id does not match"
            )

    def checkpoint(self) -> Checkpoint:
        """Export completed graph checkpoints using the native serializer."""
        from langchain_core.messages import messages_to_dict

        config = self._config()
        if self.graph.get_state(config).next:
            raise UnsupportedCapabilityError(
                "Checkpoint capture requires a completed turn; "
                "a node or interrupt is still pending"
            )
        saver = self.graph.checkpointer
        records = []
        seen = set()
        for item in saver.list(config):
            namespace = item.config["configurable"].get("checkpoint_ns", "")
            if namespace in seen:
                continue
            seen.add(namespace)
            kind, data = saver.serde.dumps_typed(item.checkpoint)
            if kind not in {"json", "msgpack"}:
                raise UnsupportedCapabilityError(
                    "Portable checkpoints require the JSON or msgpack serializer"
                )
            records.append(
                {
                    "namespace": namespace,
                    "kind": kind,
                    "data": base64.b64encode(data).decode("ascii"),
                    "metadata": item.metadata,
                }
            )
        state = self.get_state()
        if self.input_key in state:
            state[self.input_key] = messages_to_dict(state[self.input_key])
        return Checkpoint(
            adapter="langgraph",
            framework_version=version("langgraph"),
            app_id=self.app_id,
            state=state,
            payload={"records": records},
        )

    def restore(self, checkpoint: Checkpoint) -> None:
        """Import native checkpoints into a new thread, preserving channel versions."""
        self.validate_checkpoint(checkpoint)
        saver = self.graph.checkpointer
        for record in checkpoint.payload["records"]:
            if record["kind"] not in {"json", "msgpack"}:
                raise ValueError("Unsupported checkpoint serialization")
            data = saver.serde.loads_typed(
                (record["kind"], base64.b64decode(record["data"], validate=True))
            )
            config = self._config()
            config["configurable"]["checkpoint_ns"] = record["namespace"]
            saver.put(config, data, record["metadata"], data["channel_versions"])

    def inject(self, fixture: StateFixture) -> None:
        """Seed values and messages through the graph's native state writers."""
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        state = deepcopy(fixture.state)
        if fixture.history:
            if self.input_key in state:
                raise ValueError("Supply history or the messages state key, not both")
            messages = []
            for turn in fixture.history:
                if turn.role == "user":
                    messages.append(HumanMessage(content=turn.content))
                else:
                    calls = [
                        {
                            "name": c.tool_name,
                            "args": c.arguments,
                            "id": c.call_id or str(uuid.uuid4()),
                        }
                        for c in turn.tool_calls
                    ]
                    messages.append(AIMessage(content=turn.content, tool_calls=calls))
                    for call, record in zip(turn.tool_calls, calls, strict=True):
                        messages.append(
                            ToolMessage(
                                content=str(call.result),
                                tool_call_id=record["id"],
                                name=call.tool_name,
                            )
                        )
            state[self.input_key] = messages
        self.graph.update_state(self._config(), state, as_node="__start__")

    def patch_state(self, patch: dict[str, Any]) -> None:
        """Apply state updates through native reducers."""
        self.graph.update_state(self._config(), deepcopy(patch), as_node="__start__")
