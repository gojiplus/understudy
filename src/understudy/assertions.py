"""Framework-independent trace assertions."""

from typing import Any

from .trace import Trace


class AssertionHelpers:
    """Helper methods for trace assertions.

    These helpers work without importing pytest.
    """

    @staticmethod
    def assert_called(trace: Trace, tool_name: str, **kwargs: Any) -> None:
        """Assert that a tool was called with optional argument matching."""
        if not trace.called(tool_name, **kwargs):
            calls = [c.tool_name for c in trace.tool_calls]
            msg = f"Expected {tool_name} to be called"
            if kwargs:
                msg += f" with {kwargs}"
            msg += f". Actual calls: {calls}"
            raise AssertionError(msg)

    @staticmethod
    def assert_not_called(trace: Trace, tool_name: str) -> None:
        """Assert that a tool was NOT called."""
        if trace.called(tool_name):
            raise AssertionError(f"Expected {tool_name} NOT to be called, but it was")

    @staticmethod
    def assert_tool_sequence(trace: Trace, expected: list[str]) -> None:
        """Assert that tools were called in a specific sequence."""
        actual = trace.call_sequence()
        if actual != expected:
            raise AssertionError(f"Expected tool sequence {expected}, got {actual}")

    @staticmethod
    def assert_terminal_state(trace: Trace, expected: str) -> None:
        """Assert the terminal state of the conversation."""
        if trace.terminal_state != expected:
            raise AssertionError(
                f"Expected terminal state '{expected}', got '{trace.terminal_state}'"
            )
