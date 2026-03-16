"""Tests for the pytest plugin."""

import pytest

from understudy import AssertionHelpers, ToolCall, Trace, Turn
from understudy.pytest_plugin import pytest_addoption, pytest_configure


class MockConfig:
    """Mock pytest config for testing."""

    def __init__(self):
        self.markers = []

    def addinivalue_line(self, name: str, line: str):
        self.markers.append((name, line))


class MockParser:
    """Mock pytest parser for testing."""

    def __init__(self):
        self.groups = {}

    def getgroup(self, name: str):
        if name not in self.groups:
            self.groups[name] = MockGroup()
        return self.groups[name]


class MockGroup:
    """Mock parser group."""

    def __init__(self):
        self.options = []

    def addoption(self, *args, **kwargs):
        self.options.append((args, kwargs))


class TestPytestConfigure:
    def test_registers_scene_marker(self):
        config = MockConfig()
        pytest_configure(config)

        assert len(config.markers) == 1
        assert config.markers[0][0] == "markers"
        assert "scene" in config.markers[0][1]


class TestPytestAddoption:
    def test_adds_report_option(self):
        parser = MockParser()
        pytest_addoption(parser)

        group = parser.groups["understudy"]
        option_names = [opt[0][0] for opt in group.options]

        assert "--understudy-report" in option_names
        assert "--understudy-model" in option_names


class TestAssertionHelpers:
    def _make_trace(self, tool_calls: list[ToolCall] | None = None) -> Trace:
        return Trace(
            scene_id="test",
            turns=[
                Turn(role="user", content="Hello"),
                Turn(
                    role="agent",
                    content="I'll help",
                    tool_calls=tool_calls or [],
                ),
            ],
            terminal_state="completed",
        )

    def test_assert_called_passes(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="lookup_order", arguments={"order_id": "123"}),
            ]
        )
        AssertionHelpers.assert_called(trace, "lookup_order")

    def test_assert_called_with_args_passes(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="lookup_order", arguments={"order_id": "123"}),
            ]
        )
        AssertionHelpers.assert_called(trace, "lookup_order", order_id="123")

    def test_assert_called_fails(self):
        trace = self._make_trace([])
        with pytest.raises(pytest.fail.Exception):
            AssertionHelpers.assert_called(trace, "lookup_order")

    def test_assert_not_called_passes(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="other_tool", arguments={}),
            ]
        )
        AssertionHelpers.assert_not_called(trace, "lookup_order")

    def test_assert_not_called_fails(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="lookup_order", arguments={}),
            ]
        )
        with pytest.raises(pytest.fail.Exception):
            AssertionHelpers.assert_not_called(trace, "lookup_order")

    def test_assert_tool_sequence_passes(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="lookup_order", arguments={}),
                ToolCall(tool_name="create_return", arguments={}),
            ]
        )
        AssertionHelpers.assert_tool_sequence(trace, ["lookup_order", "create_return"])

    def test_assert_tool_sequence_fails(self):
        trace = self._make_trace(
            [
                ToolCall(tool_name="create_return", arguments={}),
                ToolCall(tool_name="lookup_order", arguments={}),
            ]
        )
        with pytest.raises(pytest.fail.Exception):
            AssertionHelpers.assert_tool_sequence(trace, ["lookup_order", "create_return"])

    def test_assert_terminal_state_passes(self):
        trace = self._make_trace()
        AssertionHelpers.assert_terminal_state(trace, "completed")

    def test_assert_terminal_state_fails(self):
        trace = self._make_trace()
        with pytest.raises(pytest.fail.Exception):
            AssertionHelpers.assert_terminal_state(trace, "failed")
