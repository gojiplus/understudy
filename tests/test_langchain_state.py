"""LangChain's actual agent runtime with fake model output and real middleware."""

import pytest

pytest.importorskip("langchain")
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from understudy import MockToolkit, run_turn
from understudy.langchain import LangChainApp


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_native_agent_tool_interception_and_checkpoint():
    real = []

    def write(value: str) -> str:
        """Write a value."""
        real.append(value)
        return "real"

    def app():
        model = ToolModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write", "args": {"value": "x"}, "id": "one"}],
                ),
                AIMessage(content="done"),
            ]
        )
        return LangChainApp(model, [write], app_id="writer-v1")

    def mocks():
        toolkit = MockToolkit(unmatched="error")
        toolkit.handle("write")(lambda value: {"mock": value})
        return toolkit

    first = run_turn(app, "write", mocks_factory=mocks, capture_checkpoint=True)
    assert real == []
    assert first.trace.tool_calls[0].result == '{"mock": "x"}'
    assert first.trace.turns[-1].content == "done"
    second = run_turn(app, "again", checkpoint=first.checkpoint, mocks_factory=mocks)
    assert real == []
    assert len(second.trace.tool_calls) == 1
    assert len(second.state_after["messages"]) > len(first.state_after["messages"])


def test_async_tool_and_mock_return_values():
    real = []

    async def write(value: str) -> str:
        """Write a value."""
        real.append(value)
        return "real"

    def app():
        return LangChainApp(
            ToolModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "write", "args": {"value": "x"}, "id": "one"}
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            ),
            [write],
        )

    def mocks():
        toolkit = MockToolkit()

        async def mocked(value):
            return {"async": value}

        toolkit.handle("write")(mocked)
        return toolkit

    result = run_turn(app, "write", mocks_factory=mocks)
    assert real == []
    assert result.trace.tool_calls[0].result == '{"async": "x"}'
    result = run_turn(app, "write")
    assert real == ["x"]
    assert result.trace.tool_calls[0].result == "real"
