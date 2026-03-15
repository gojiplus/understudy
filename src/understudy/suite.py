"""Suite: run a collection of scenes and produce aggregate results."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .check import CheckResult, check
from .models import Scene
from .runner import AgentApp, run
from .trace import Trace

if TYPE_CHECKING:
    from .storage import RunStorage


@dataclass
class SceneResult:
    """Result of running a single scene."""

    scene_id: str
    trace: Trace
    check_result: CheckResult
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.check_result.passed


@dataclass
class SuiteResults:
    """Aggregate results from running a suite of scenes."""

    results: list[SceneResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def failed(self) -> list[SceneResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        lines = [f"{self.pass_count}/{len(self.results)} passed"]
        for r in self.failed:
            if r.error:
                lines.append(f"  FAILED: {r.scene_id} (error: {r.error})")
            else:
                for c in r.check_result.failed_checks:
                    lines.append(f"  FAILED: {r.scene_id} ({c.label}: {c.detail})")
        return "\n".join(lines)

    def to_junit_xml(self, path: str | Path) -> None:
        """Export results as JUnit XML for CI integration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        testsuite = ET.Element(
            "testsuite",
            name="understudy",
            tests=str(len(self.results)),
            failures=str(self.fail_count),
            errors="0",
        )

        for r in self.results:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                name=r.scene_id,
                classname="understudy",
            )
            if r.trace.duration:
                testcase.set("time", f"{r.trace.duration.total_seconds():.2f}")

            if not r.passed:
                if r.error:
                    ET.SubElement(testcase, "failure", message=r.error, type="error")
                else:
                    msgs = [f"{c.label}: {c.detail}" for c in r.check_result.failed_checks]
                    ET.SubElement(
                        testcase,
                        "failure",
                        message="; ".join(msgs),
                        type="assertion",
                    )

        tree = ET.ElementTree(testsuite)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=True)


class Suite:
    """A collection of scenes to run as a test suite."""

    def __init__(self, scenes: list[Scene]):
        self.scenes = scenes

    @classmethod
    def from_directory(cls, path: str | Path) -> Suite:
        """Load all .yaml and .json scene files from a directory."""
        path = Path(path)
        scenes = []
        for f in sorted(path.iterdir()):
            if f.suffix in (".yaml", ".yml", ".json"):
                try:
                    scenes.append(Scene.from_file(f))
                except Exception as e:
                    raise ValueError(f"Failed to load scene from {f}: {e}") from e
        return cls(scenes)

    def run(
        self,
        app: AgentApp,
        parallel: int = 1,
        storage: RunStorage | None = None,
        tags: dict[str, str] | None = None,
        n_sims: int = 1,
        **run_kwargs: Any,
    ) -> SuiteResults:
        """Run all scenes and return aggregate results.

        Args:
            app: The agent application to test.
            parallel: Number of scenes to run in parallel (default: 1).
            storage: Optional RunStorage to persist each scene run.
            tags: Optional dict of tags for filtering and comparison.
            n_sims: Number of simulations per scene (default: 1).
            **run_kwargs: Additional kwargs passed to understudy.run().

        Returns:
            SuiteResults with individual scene outcomes.
        """
        results = SuiteResults()

        sim_tasks = []
        for scene in self.scenes:
            for sim_index in range(n_sims):
                sim_tasks.append((scene, sim_index))

        if parallel <= 1:
            for scene, sim_index in sim_tasks:
                result = self._run_scene(
                    app, scene, storage=storage, tags=tags, sim_index=sim_index, **run_kwargs
                )
                results.results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {
                    executor.submit(
                        self._run_scene,
                        app,
                        scene,
                        storage=storage,
                        tags=tags,
                        sim_index=sim_index,
                        **run_kwargs,
                    ): (scene, sim_index)
                    for scene, sim_index in sim_tasks
                }
                for future in as_completed(futures):
                    results.results.append(future.result())

        return results

    def _run_scene(
        self,
        app: AgentApp,
        scene: Scene,
        storage: RunStorage | None = None,
        tags: dict[str, str] | None = None,
        sim_index: int = 0,
        **run_kwargs: Any,
    ) -> SceneResult:
        """Run a single scene and check expectations."""
        scene_id_with_index = f"{scene.id}" if sim_index == 0 else f"{scene.id}_{sim_index}"
        try:
            trace = run(app, scene, **run_kwargs)
            result = check(trace, scene.expectations)
            scene_result = SceneResult(
                scene_id=scene_id_with_index,
                trace=trace,
                check_result=result,
            )
            if storage:
                storage.save(trace, scene, check_result=result, tags=tags)
            return scene_result
        except Exception as e:
            return SceneResult(
                scene_id=scene_id_with_index,
                trace=Trace(scene_id=scene.id),
                check_result=CheckResult(),
                error=str(e),
            )
