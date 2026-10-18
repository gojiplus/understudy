"""Tests for the pytest plugin."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from understudy import AssertionHelpers, ToolCall, Trace, Turn
from understudy.pytest_plugin import pytest_addoption, pytest_configure

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "keyword"])
def test_scene_fixture_loads_relative_to_test(tmp_path, path_kind):
    from understudy.pytest_plugin import scene

    path = tmp_path / "scene.yaml"
    path.write_text(
        "id: example\nstarting_prompt: hello\n"
        "conversation_plan: say goodbye\npersona: cooperative\n"
    )
    name = str(path) if path_kind == "absolute" else "scene.yaml"
    marker = (
        pytest.mark.scene(path=name)
        if path_kind == "keyword"
        else pytest.mark.scene(name)
    )
    request = SimpleNamespace(
        path=tmp_path / "test_agent.py",
        node=Mock(get_closest_marker=Mock(return_value=marker)),
    )
    assert scene.__wrapped__(request).id == "example"


@pytest.mark.parametrize(
    ("marker", "error"),
    [
        (pytest.mark.scene(), "requires a path"),
        (pytest.mark.scene("missing.yaml"), "not found"),
    ],
)
def test_scene_fixture_errors(tmp_path, marker, error):
    from understudy.pytest_plugin import scene

    request = SimpleNamespace(
        path=tmp_path / "test_agent.py",
        node=Mock(get_closest_marker=Mock(return_value=marker)),
    )
    with pytest.raises(pytest.fail.Exception, match=error):
        scene.__wrapped__(request)


def test_fixtures_without_marker():
    from understudy.pytest_plugin import scene, suite_results, trace

    request = SimpleNamespace(node=Mock(get_closest_marker=Mock(return_value=None)))
    assert scene.__wrapped__(request) is None
    assert trace.__wrapped__(request, None, None, None) is None
    assert suite_results.__wrapped__(request, None, None) is None
    request.node.get_closest_marker.return_value = pytest.mark.scene()
    assert suite_results.__wrapped__(request, None, None) is None


def test_trace_fixture_forwards_model_and_reports_errors(monkeypatch):
    import understudy.pytest_plugin as plugin

    request = SimpleNamespace(config=Mock(getoption=Mock(return_value="test-model")))
    run = Mock(return_value="trace")
    monkeypatch.setattr(plugin, "run", run)
    assert plugin.trace.__wrapped__(request, "app", "mocks", "scene") == "trace"
    run.assert_called_once_with(
        app="app", mocks="mocks", scene="scene", simulator_model="test-model"
    )
    run.side_effect = ValueError("agent failed")
    with pytest.raises(
        pytest.fail.Exception, match="Scene execution failed: agent failed"
    ):
        plugin.trace.__wrapped__(request, "app", "mocks", "scene")


@pytest.mark.parametrize("directory", [False, True])
def test_suite_fixture(tmp_path, monkeypatch, directory):
    import understudy.pytest_plugin as plugin

    path = tmp_path / "scene.yaml"
    path.write_text(
        "id: example\nstarting_prompt: hello\n"
        "conversation_plan: say goodbye\npersona: cooperative\n"
    )
    marker = pytest.mark.scene(str(tmp_path) if directory else "scene.yaml")
    request = SimpleNamespace(
        path=tmp_path / "test_agent.py",
        node=Mock(get_closest_marker=Mock(return_value=marker)),
        config=Mock(getoption=Mock(return_value="model")),
    )
    run = Mock(return_value="results")
    monkeypatch.setattr(plugin.Suite, "run", run)
    assert plugin.suite_results.__wrapped__(request, "app", "mocks") == "results"
    run.assert_called_once_with(app="app", mocks="mocks", simulator_model="model")


def test_installed_plugin_discovery(pytester):
    pytester.makeini("[pytest]\naddopts = --strict-markers\n")
    pytester.makepyfile("""
        import pytest

        def test_options(pytestconfig):
            assert pytestconfig.getoption("understudy_model") == "offline"

        @pytest.mark.scene(path="scene.yaml")
        def test_scene_loading(scene):
            assert scene.id == "example"
    """)
    (pytester.path / "scene.yaml").write_text(
        "id: example\nstarting_prompt: hello\n"
        "conversation_plan: say goodbye\npersona: cooperative\n"
    )
    pytester.runpytest_subprocess("--understudy-model=offline", "-q").assert_outcomes(
        passed=2
    )


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
            AssertionHelpers.assert_tool_sequence(
                trace, ["lookup_order", "create_return"]
            )

    def test_assert_terminal_state_passes(self):
        trace = self._make_trace()
        AssertionHelpers.assert_terminal_state(trace, "completed")

    def test_assert_terminal_state_fails(self):
        trace = self._make_trace()
        with pytest.raises(pytest.fail.Exception):
            AssertionHelpers.assert_terminal_state(trace, "failed")


def test_turn_runner_fixture(pytester):
    pytester.makepyfile("""
import pytest
from understudy import AdapterCapabilities, AgentResponse
class App:
    capabilities = AdapterCapabilities(tools=True, agents=True)
    def start(self, mocks=None): pass
    def send(self, message): return AgentResponse(message)
    def stop(self): pass
@pytest.fixture
def app_factory(): return App
def test_one_turn(turn_runner):
    result = turn_runner("hello")
    assert [t.content for t in result.trace.turns] == ["hello", "hello"]
""")
    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=1)
