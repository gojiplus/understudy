"""Storage: persist simulation runs to disk."""

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from .check import CheckResult
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
        """Get aggregate summary of all runs."""
        runs = self.load_all()
        n = len(runs)
        if n == 0:
            return self._empty_summary()

        passed = sum(1 for r in runs if r.get("metadata", {}).get("passed"))
        total_turns = sum(r.get("metadata", {}).get("turn_count", 0) for r in runs)

        tool_counts: dict[str, int] = {}
        terminal_counts: dict[str, int] = {}
        agent_counts: dict[str, int] = {}

        for r in runs:
            meta = r.get("metadata", {})
            for tool in meta.get("tools_called", []):
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            state = meta.get("terminal_state", "unknown")
            terminal_counts[state] = terminal_counts.get(state, 0) + 1
            for agent in meta.get("agents_invoked", []):
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        perf = self._aggregate_performance(runs)
        t = perf["turns"]

        return {
            "total_runs": n,
            "pass_rate": passed / n,
            "avg_turns": total_turns / n,
            "tool_usage": tool_counts,
            "terminal_states": terminal_counts,
            "agents": agent_counts,
            "total_tokens": perf["tokens"],
            "total_input_tokens": perf["input"],
            "total_output_tokens": perf["output"],
            "total_thinking_tokens": perf["thinking"],
            "total_latency_ms": perf["latency"],
            "avg_tokens_per_run": perf["tokens"] / n,
            "avg_latency_per_run_ms": perf["latency"] / n,
            "avg_latency_per_turn_ms": perf["latency"] / t if t else 0,
            "avg_input_tokens_per_turn": perf["input"] / t if t else 0,
            "avg_output_tokens_per_turn": perf["output"] / t if t else 0,
            "avg_thinking_tokens_per_turn": perf["thinking"] / t if t else 0,
            "judge_stats": self._compute_judge_stats(runs),
        }

    def _empty_summary(self) -> dict[str, Any]:
        return {
            "total_runs": 0,
            "pass_rate": 0.0,
            "avg_turns": 0.0,
            "tool_usage": {},
            "terminal_states": {},
            "agents": {},
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_thinking_tokens": 0,
            "total_latency_ms": 0,
            "avg_tokens_per_run": 0,
            "avg_latency_per_run_ms": 0,
            "avg_latency_per_turn_ms": 0,
            "avg_input_tokens_per_turn": 0,
            "avg_output_tokens_per_turn": 0,
            "avg_thinking_tokens_per_turn": 0,
            "judge_stats": {},
        }

    def _aggregate_performance(self, runs: list[dict[str, Any]]) -> dict[str, int]:
        tokens = input_tokens = output_tokens = thinking_tokens = latency = turns = 0
        for r in runs:
            trace = r.get("trace")
            if trace and hasattr(trace, "metrics") and trace.metrics:
                m = trace.metrics
                tokens += m.total_tokens
                input_tokens += m.total_input_tokens
                output_tokens += m.total_output_tokens
                thinking_tokens += m.total_thinking_tokens
                latency += m.agent_time_ms
                turns += len(m.turns)
        return {
            "tokens": tokens,
            "input": input_tokens,
            "output": output_tokens,
            "thinking": thinking_tokens,
            "latency": latency,
            "turns": turns,
        }

    def _compute_judge_stats(self, runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Compute aggregate judge statistics across all runs."""
        rubric_data: dict[str, dict[str, list]] = {}

        for run in runs:
            judges = run.get("judges", {})
            for name, result in judges.items():
                if name not in rubric_data:
                    rubric_data[name] = {"scores": [], "agreements": []}

                score = result.get("score")
                if score is not None:
                    rubric_data[name]["scores"].append(score)

                agreement = result.get("agreement_rate")
                if agreement is not None:
                    rubric_data[name]["agreements"].append(agreement)

        result = {}
        for name, data in rubric_data.items():
            scores = data["scores"]
            agreements = data["agreements"]
            result[name] = {
                "pass_rate": sum(scores) / len(scores) if scores else 0.0,
                "avg_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
                "count": len(scores),
            }

        return result


class TraceStorage:
    """Persist simulation traces to disk (without evaluation results).

    Used by simulate/simulate_batch for simulation-only workflows.
    """

    def __init__(self, path: Path | str = ".understudy/traces"):
        self.path = Path(path)

    def save(
        self,
        trace: Trace,
        scene: Scene,
        sim_index: int = 0,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Save a trace and return the trace_id.

        Args:
            trace: The execution trace.
            scene: The scene that was run.
            sim_index: Index of this simulation (for n_sims > 1).
            tags: Optional metadata tags.

        Returns:
            The trace_id (can be used to load the trace later).
        """
        self.path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_id = f"{trace.scene_id}_{sim_index}_{timestamp}"
        trace_file = self.path / f"{trace_id}.json"

        data = {
            "trace": json.loads(trace.model_dump_json()),
            "scene": json.loads(scene.model_dump_json()),
            "metadata": {
                "trace_id": trace_id,
                "scene_id": trace.scene_id,
                "sim_index": sim_index,
                "timestamp": datetime.now().isoformat(),
                "tags": tags or {},
            },
        }

        trace_file.write_text(json.dumps(data, indent=2, default=str))
        return trace_id

    def load(self, trace_id: str) -> dict[str, Any]:
        """Load a trace by its ID.

        Args:
            trace_id: The trace identifier.

        Returns:
            Dict containing trace, scene, and metadata.
        """
        trace_file = self.path / f"{trace_id}.json"
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace not found: {trace_id}")

        data = json.loads(trace_file.read_text())
        data["trace"] = Trace.model_validate(data["trace"])
        data["scene"] = Scene.model_validate(data["scene"])
        return data

    def load_trace(self, trace_id: str) -> Trace:
        """Load just the trace object by its ID."""
        return self.load(trace_id)["trace"]

    def list_traces(self) -> list[str]:
        """List all trace IDs in storage."""
        if not self.path.exists():
            return []

        traces = []
        for f in self.path.iterdir():
            if f.suffix == ".json" and f.is_file():
                traces.append(f.stem)

        return sorted(traces, reverse=True)

    def load_all(self) -> list[dict[str, Any]]:
        """Load all traces."""
        return [self.load(trace_id) for trace_id in self.list_traces()]

    def delete(self, trace_id: str) -> None:
        """Delete a trace by its ID."""
        trace_file = self.path / f"{trace_id}.json"
        if trace_file.exists():
            trace_file.unlink()

    def clear(self) -> None:
        """Delete all traces."""
        if self.path.exists():
            import shutil

            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)


class EvaluationStorage:
    """Persist evaluation results to disk.

    Used by evaluate/evaluate_batch for storing evaluation results.
    """

    def __init__(self, path: Path | str = ".understudy/results"):
        self.path = Path(path)

    def save(
        self,
        trace_id: str,
        check_result: CheckResult,
        judges: dict[str, Any] | None = None,
    ) -> str:
        """Save evaluation results.

        Args:
            trace_id: The trace that was evaluated.
            check_result: The evaluation result.
            judges: Optional judge results.

        Returns:
            The result filename.
        """
        self.path.mkdir(parents=True, exist_ok=True)

        result_id = f"{trace_id}_eval"
        result_file = self.path / f"{result_id}.json"

        data = {
            "trace_id": trace_id,
            "passed": check_result.passed,
            "checks": [
                {"label": c.label, "passed": c.passed, "detail": c.detail}
                for c in check_result.checks
            ],
            "metrics": {
                name: {
                    "name": m.name,
                    "value": m.value,
                    "passed": m.passed,
                    "detail": m.detail,
                }
                for name, m in check_result.metrics.items()
            },
        }

        if judges:
            judges_data = {}
            for name, result in judges.items():
                if hasattr(result, "model_dump"):
                    judges_data[name] = result.model_dump()
                elif hasattr(result, "__dict__"):
                    judges_data[name] = result.__dict__
                else:
                    judges_data[name] = result
            data["judges"] = judges_data

        result_file.write_text(json.dumps(data, indent=2, default=str))
        return result_id

    def load(self, result_id: str) -> dict[str, Any]:
        """Load evaluation results by ID."""
        result_file = self.path / f"{result_id}.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Result not found: {result_id}")

        return json.loads(result_file.read_text())

    def list_results(self) -> list[str]:
        """List all result IDs in storage."""
        if not self.path.exists():
            return []

        results = []
        for f in self.path.iterdir():
            if f.suffix == ".json" and f.is_file():
                results.append(f.stem)

        return sorted(results, reverse=True)

    def load_all(self) -> list[dict[str, Any]]:
        """Load all results."""
        return [self.load(result_id) for result_id in self.list_results()]

    def delete(self, result_id: str) -> None:
        """Delete a result by its ID."""
        result_file = self.path / f"{result_id}.json"
        if result_file.exists():
            result_file.unlink()

    def clear(self) -> None:
        """Delete all results."""
        if self.path.exists():
            import shutil

            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
