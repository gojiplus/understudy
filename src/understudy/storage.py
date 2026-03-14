"""Storage: persist simulation runs to disk."""

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Scene
from .trace import Trace


class RunStorage:
    """Persist simulation runs to disk for later analysis and reporting."""

    def __init__(self, path: Path | str = ".understudy/runs"):
        """
        Args:
            path: Directory to store run data.
        """
        self.path = Path(path)

    def save(
        self,
        trace: Trace,
        scene: Scene,
        judges: dict[str, Any] | None = None,
        check_result: Any | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save a run and return the run_id.

        Args:
            trace: The execution trace.
            scene: The scene that was run.
            judges: Optional dict of judge results.
            check_result: Optional CheckResult from expectations validation.
            tags: Optional dict of tags for filtering and comparison.

        Returns:
            The run_id (can be used to load the run later).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        run_id = f"{trace.scene_id}_{timestamp}_{suffix}"
        run_dir = self.path / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "trace.json").write_text(trace.model_dump_json(indent=2))
        (run_dir / "scene.json").write_text(scene.model_dump_json(indent=2))

        if judges:
            judges_data = {}
            for name, result in judges.items():
                if hasattr(result, "model_dump"):
                    judges_data[name] = result.model_dump()
                elif hasattr(result, "__dict__"):
                    judges_data[name] = result.__dict__
                else:
                    judges_data[name] = result
            (run_dir / "judges.json").write_text(json.dumps(judges_data, indent=2, default=str))

        if check_result:
            check_data = {
                "passed": check_result.passed,
                "checks": [
                    {"label": c.label, "passed": c.passed, "detail": c.detail}
                    for c in check_result.checks
                ],
            }
            (run_dir / "check.json").write_text(json.dumps(check_data, indent=2))

        metadata = {
            "run_id": run_id,
            "scene_id": trace.scene_id,
            "timestamp": timestamp,
            "passed": check_result.passed if check_result else None,
            "terminal_state": trace.terminal_state,
            "turn_count": trace.turn_count,
            "tools_called": trace.call_sequence(),
            "agents_invoked": trace.agents_invoked(),
            "tags": tags or {},
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return run_id

    def load(self, run_id: str) -> dict[str, Any]:
        """Load a run by its ID.

        Args:
            run_id: The run identifier.

        Returns:
            Dict containing trace, scene, judges, check, and metadata.
        """
        run_dir = self.path / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")

        result: dict[str, Any] = {"run_id": run_id}

        trace_file = run_dir / "trace.json"
        if trace_file.exists():
            result["trace"] = Trace.model_validate_json(trace_file.read_text())

        scene_file = run_dir / "scene.json"
        if scene_file.exists():
            result["scene"] = Scene.model_validate_json(scene_file.read_text())

        judges_file = run_dir / "judges.json"
        if judges_file.exists():
            result["judges"] = json.loads(judges_file.read_text())

        check_file = run_dir / "check.json"
        if check_file.exists():
            result["check"] = json.loads(check_file.read_text())

        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            result["metadata"] = json.loads(metadata_file.read_text())

        return result

    def list_runs(self) -> list[str]:
        """List all run IDs in storage.

        Returns:
            List of run IDs, sorted by timestamp (newest first).
        """
        if not self.path.exists():
            return []

        runs = []
        for run_dir in self.path.iterdir():
            if run_dir.is_dir() and (run_dir / "metadata.json").exists():
                runs.append(run_dir.name)

        return sorted(runs, reverse=True)

    def delete(self, run_id: str) -> None:
        """Delete a run by its ID.

        Args:
            run_id: The run identifier.
        """
        run_dir = self.path / run_id
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)

    def clear(self) -> None:
        """Delete all runs."""
        if self.path.exists():
            import shutil

            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[dict[str, Any]]:
        """Load all runs.

        Returns:
            List of run data dicts.
        """
        return [self.load(run_id) for run_id in self.list_runs()]

    def get_summary(self) -> dict[str, Any]:
        """Get aggregate summary of all runs.

        Returns:
            Summary statistics including pass rate, tool usage, etc.
        """
        runs = self.load_all()
        if not runs:
            return {
                "total_runs": 0,
                "pass_rate": 0.0,
                "avg_turns": 0.0,
                "tool_usage": {},
                "terminal_states": {},
                "agents": {},
            }

        passed = sum(1 for r in runs if r.get("metadata", {}).get("passed"))
        total_turns = sum(r.get("metadata", {}).get("turn_count", 0) for r in runs)

        tool_counts: dict[str, int] = {}
        for r in runs:
            for tool in r.get("metadata", {}).get("tools_called", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        terminal_counts: dict[str, int] = {}
        for r in runs:
            state = r.get("metadata", {}).get("terminal_state", "unknown")
            terminal_counts[state] = terminal_counts.get(state, 0) + 1

        agent_counts: dict[str, int] = {}
        for r in runs:
            for agent in r.get("metadata", {}).get("agents_invoked", []):
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "total_runs": len(runs),
            "pass_rate": passed / len(runs) if runs else 0.0,
            "avg_turns": total_turns / len(runs) if runs else 0.0,
            "tool_usage": tool_counts,
            "terminal_states": terminal_counts,
            "agents": agent_counts,
        }
