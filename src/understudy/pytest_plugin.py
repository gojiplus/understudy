"""Pytest plugin for understudy.

This plugin provides native pytest integration for understudy scene testing.

Usage:
    # conftest.py
    pytest_plugins = ["understudy.pytest"]

    @pytest.fixture
    def app():
        return MyAgentApp()

    @pytest.fixture
    def mocks():
        return my_mocks()

    # test_agent.py
    @pytest.mark.scene("scenes/return_request.yaml")
    def test_return_flow(trace):
        assert trace.called("lookup_order")
        assert trace.called("create_return")

CLI options:
    --understudy-report=path.html   Generate HTML report
    --understudy-model=MODEL        Override simulator model
"""

from pathlib import Path
from typing import Any

import pytest

from .models import Scene
from .runner import run
from .suite import Suite
from .trace import Trace


def pytest_configure(config: pytest.Config) -> None:
    """Register the scene marker."""
    config.addinivalue_line(
        "markers",
        "scene(path): mark test to run with an understudy scene file",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add understudy CLI options."""
    group = parser.getgroup("understudy")
    group.addoption(
        "--understudy-report",
        action="store",
        dest="understudy_report",
        default=None,
        help="Path to generate HTML report",
    )
    group.addoption(
        "--understudy-model",
        action="store",
        dest="understudy_model",
        default="gpt-4o",
        help="Simulator model to use (default: gpt-4o)",
    )


class SceneTestItem:
    """Holds data for a scene-based test."""

    def __init__(
        self,
        scene: Scene,
        trace: Trace | None = None,
        error: str | None = None,
    ):
        self.scene = scene
        self.trace = trace
        self.error = error


@pytest.fixture
def scene(request: pytest.FixtureRequest) -> Scene | None:
    """Fixture that provides the Scene object from the @pytest.mark.scene decorator.

    Returns None if the test doesn't have the scene marker.
    """
    marker = request.node.get_closest_marker("scene")
    if marker is None:
        return None

    scene_path = marker.args[0] if marker.args else marker.kwargs.get("path")
    if scene_path is None:
        pytest.fail("@pytest.mark.scene requires a path argument")

    path = Path(scene_path)
    if not path.is_absolute():
        test_dir = Path(request.fspath).parent
        path = test_dir / path

    if not path.exists():
        pytest.fail(f"Scene file not found: {path}")

    return Scene.from_file(path)


@pytest.fixture
def trace(request: pytest.FixtureRequest, app, mocks, scene: Scene | None) -> Trace | None:
    """Fixture that runs the scene and provides the execution trace.

    This fixture:
    1. Loads the scene from the @pytest.mark.scene path
    2. Runs the simulation with the app and mocks fixtures
    3. Returns the Trace for assertions

    Requires app and mocks fixtures to be defined in conftest.py.
    """
    if scene is None:
        return None

    model = request.config.getoption("understudy_model", "gpt-4o")

    try:
        trace_result = run(
            app=app,
            scene=scene,
            mocks=mocks,
            simulator_model=model,
        )
        return trace_result
    except Exception as e:
        pytest.fail(f"Scene execution failed: {e}")


@pytest.fixture
def suite_results(request: pytest.FixtureRequest, app, mocks) -> Any:
    """Fixture to run multiple scenes as a suite.

    Use this fixture when you want to run all scenes in a directory:

        def test_all_scenes(suite_results):
            assert suite_results.all_passed
    """
    marker = request.node.get_closest_marker("scene")
    if marker is None:
        return None

    scene_path = marker.args[0] if marker.args else marker.kwargs.get("path")
    if scene_path is None:
        return None

    path = Path(scene_path)
    if not path.is_absolute():
        test_dir = Path(request.fspath).parent
        path = test_dir / path

    if path.is_dir():
        suite = Suite.from_directory(path)
    else:
        scene = Scene.from_file(path)
        suite = Suite([scene])

    model = request.config.getoption("understudy_model", "gpt-4o")

    return suite.run(
        app=app,
        mocks=mocks,
        simulator_model=model,
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Process scene markers on collected tests."""
    for item in items:
        marker = item.get_closest_marker("scene")
        if marker:
            pass


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Generate report at end of session if requested."""
    report_path = session.config.getoption("understudy_report")
    if report_path:
        pass


class AssertionHelpers:
    """Helper methods for trace assertions.

    These are available on the trace object for convenient assertions.
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
            pytest.fail(msg)

    @staticmethod
    def assert_not_called(trace: Trace, tool_name: str) -> None:
        """Assert that a tool was NOT called."""
        if trace.called(tool_name):
            pytest.fail(f"Expected {tool_name} NOT to be called, but it was")

    @staticmethod
    def assert_tool_sequence(trace: Trace, expected: list[str]) -> None:
        """Assert that tools were called in a specific sequence."""
        actual = trace.call_sequence()
        if actual != expected:
            pytest.fail(f"Expected tool sequence {expected}, got {actual}")

    @staticmethod
    def assert_terminal_state(trace: Trace, expected: str) -> None:
        """Assert the terminal state of the conversation."""
        if trace.terminal_state != expected:
            pytest.fail(f"Expected terminal state '{expected}', got '{trace.terminal_state}'")
