"""Tool definitions for the evaluator agent.

These are the tools the evaluator agent uses to interact
with understudy's primitives.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..check import CheckResult, check
from ..mocks import MockToolkit
from ..models import Scene
from ..runner import AgentApp, run
from ..trace import Trace
from .analyzer import AnalysisResult, DeepAnalyzer
from .coverage import CoverageTracker
from .generator import GenerationMode, SceneGenerator
from .inputs import AgentSpec


class ToolResult:
    """Base result from a tool."""

    def __init__(self, success: bool, data: Any = None, error: str | None = None):
        self.success = success
        self.data = data
        self.error = error


@dataclass
class SceneExecutionResult:
    """Result from running a scene."""

    scene_id: str
    trace: Trace
    check_result: CheckResult
    passed: bool
    error: str | None = None


class EvaluatorToolkit:
    """Tools available to the evaluator agent.

    Wraps understudy primitives with a consistent interface.
    """

    def __init__(
        self,
        target_app: AgentApp,
        spec: AgentSpec,
        mocks: MockToolkit | None = None,
        model: str = "claude-sonnet-4-20250514",
        simulator_model: str = "gpt-4o",
    ):
        """Initialize the toolkit.

        Args:
            target_app: The agent application to evaluate
            spec: The agent specification
            mocks: Optional mock toolkit for tool responses
            model: Model for LLM operations (generation, analysis)
            simulator_model: Model for the user simulator
        """
        self.target_app = target_app
        self.spec = spec
        self.mocks = mocks
        self.model = model
        self.simulator_model = simulator_model

        self.generator = SceneGenerator(model=model)
        self.analyzer = DeepAnalyzer(model=model)
        self.coverage_tracker = CoverageTracker(spec)

        self._execution_history: list[SceneExecutionResult] = []

    def generate_scenes(
        self,
        mode: GenerationMode = GenerationMode.SYSTEMATIC,
        capability_ids: list[str] | None = None,
        max_scenes: int = 10,
    ) -> ToolResult:
        """Generate test scenes.

        Args:
            mode: Generation mode (systematic, adversarial, focused)
            capability_ids: Specific capabilities to test (for focused mode)
            max_scenes: Maximum scenes to generate

        Returns:
            ToolResult with list of generated Scene objects
        """
        try:
            scenes = self.generator.generate(
                spec=self.spec,
                mode=mode,
                capability_ids=capability_ids,
                max_scenes=max_scenes,
                existing_scenes=[],
            )
            return ToolResult(success=True, data=scenes)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def run_scene(self, scene: Scene) -> ToolResult:
        """Execute a scene against the target agent.

        Args:
            scene: The scene to run

        Returns:
            ToolResult with SceneExecutionResult
        """
        try:
            trace = run(
                app=self.target_app,
                scene=scene,
                mocks=self.mocks,
                simulator_model=self.simulator_model,
            )

            check_result = check(trace, scene.expectations)

            result = SceneExecutionResult(
                scene_id=scene.id,
                trace=trace,
                check_result=check_result,
                passed=check_result.passed,
            )

            self._execution_history.append(result)
            self.coverage_tracker.record_scene(scene, trace, check_result.passed)

            return ToolResult(success=True, data=result)
        except Exception as e:
            import traceback

            result = SceneExecutionResult(
                scene_id=scene.id,
                trace=Trace(scene_id=scene.id),
                check_result=CheckResult(),
                passed=False,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )
            self._execution_history.append(result)
            return ToolResult(success=False, data=result, error=str(e))

    def run_scenes_batch(
        self,
        scenes: list[Scene],
        parallel: int = 1,
    ) -> ToolResult:
        """Run multiple scenes.

        Args:
            scenes: List of scenes to run
            parallel: Number of parallel executions (currently sequential)

        Returns:
            ToolResult with list of SceneExecutionResults
        """
        results = []
        for scene in scenes:
            result = self.run_scene(scene)
            if result.data:
                results.append(result.data)

        return ToolResult(success=True, data=results)

    def analyze_trace(
        self,
        trace: Trace,
        check_result: CheckResult,
        scene_context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Deeply analyze a trace.

        Args:
            trace: The execution trace
            check_result: The check result
            scene_context: Optional context from the scene

        Returns:
            ToolResult with AnalysisResult
        """
        try:
            result = self.analyzer.analyze_trace(
                trace=trace,
                check_result=check_result,
                spec=self.spec,
                scene_context=scene_context,
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def analyze_patterns(
        self,
        results: list[AnalysisResult] | None = None,
    ) -> ToolResult:
        """Analyze patterns across failures.

        Args:
            results: List of analysis results (uses history if not provided)

        Returns:
            ToolResult with PatternAnalysis
        """
        try:
            if results is None:
                results = []
                for exec_result in self._execution_history:
                    if not exec_result.passed:
                        analysis = self.analyzer.analyze_trace(
                            trace=exec_result.trace,
                            check_result=exec_result.check_result,
                            spec=self.spec,
                        )
                        results.append(analysis)

            pattern_analysis = self.analyzer.analyze_patterns(results, self.spec)
            return ToolResult(success=True, data=pattern_analysis)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_coverage(self) -> ToolResult:
        """Get current test coverage.

        Returns:
            ToolResult with CoverageReport
        """
        report = self.coverage_tracker.get_report()
        return ToolResult(success=True, data=report)

    def suggest_next_tests(self, max_suggestions: int = 5) -> ToolResult:
        """Get suggestions for what to test next.

        Args:
            max_suggestions: Maximum number of suggestions

        Returns:
            ToolResult with list of test suggestions
        """
        suggestions = self.coverage_tracker.suggest_next_tests(max_suggestions)
        return ToolResult(success=True, data=suggestions)

    def save_scenes(
        self,
        scenes: list[Scene],
        output_dir: Path | str,
    ) -> ToolResult:
        """Save generated scenes to disk.

        Args:
            scenes: Scenes to save
            output_dir: Directory to save to

        Returns:
            ToolResult with list of saved paths
        """
        try:
            paths = self.generator.save_scenes(scenes, Path(output_dir))
            return ToolResult(success=True, data=paths)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_execution_history(self) -> list[SceneExecutionResult]:
        """Get all execution results from this session."""
        return self._execution_history

    def get_failed_executions(self) -> list[SceneExecutionResult]:
        """Get only failed execution results."""
        return [r for r in self._execution_history if not r.passed]

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history = []
