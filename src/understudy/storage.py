"""Storage: persist simulation runs to disk."""

import json
import secrets
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from .check import CheckResult
from .models import Scene
from .trace import Trace


class FileStorage[T](ABC):
    """Base class for JSON file storage.

    Provides common file I/O operations for storing and retrieving JSON data.
    Subclasses implement the specifics of how data is structured and serialized.
    """

    def __init__(self, path: Path | str, use_subdirs: bool = False):
        """
        Args:
            path: Directory to store data.
            use_subdirs: If True, store each item in its own subdirectory.
                        If False, store as individual JSON files.
        """
        self.path = Path(path)
        self.use_subdirs = use_subdirs

    @abstractmethod
    def _generate_key(self, **kwargs: Any) -> str:
        """Generate a unique key for storing the data."""
        ...

    @abstractmethod
    def _serialize(self, key: str, **kwargs: Any) -> dict[str, Any]:
        """Serialize the data for storage."""
        ...

    @abstractmethod
    def _deserialize(self, key: str, stored: dict[str, Any]) -> T:
        """Deserialize stored data back to its original form."""
        ...

    def _item_path(self, key: str) -> Path:
        """Get the path for storing an item."""
        if self.use_subdirs:
            return self.path / key
        return self.path / f"{key}.json"

    def _save_internal(self, **kwargs: Any) -> str:
        """Internal save implementation used by subclasses."""
        self.path.mkdir(parents=True, exist_ok=True)
        key = self._generate_key(**kwargs)
        serialized = self._serialize(key, **kwargs)

        if self.use_subdirs:
            item_dir = self._item_path(key)
            item_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in serialized.items():
                (item_dir / filename).write_text(
                    json.dumps(content, indent=2, default=str)
                    if not isinstance(content, str)
                    else content
                )
        else:
            item_file = self._item_path(key)
            item_file.write_text(json.dumps(serialized, indent=2, default=str))

        return key

    def load(self, key: str) -> T:
        """Load data by its key.

        Args:
            key: The identifier.

        Returns:
            The deserialized data.
        """
        item_path = self._item_path(key)
        if self.use_subdirs:
            if not item_path.exists():
                raise FileNotFoundError(f"Item not found: {key}")
            stored = {}
            for f in item_path.iterdir():
                if f.suffix == ".json":
                    stored[f.name] = json.loads(f.read_text())
        else:
            if not item_path.exists():
                raise FileNotFoundError(f"Item not found: {key}")
            stored = json.loads(item_path.read_text())

        return self._deserialize(key, stored)

    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of keys, sorted by name (newest first for timestamp-based keys).
        """
        if not self.path.exists():
            return []

        keys = []
        for item in self.path.iterdir():
            if self.use_subdirs:
                if item.is_dir():
                    keys.append(item.name)
            else:
                if item.suffix == ".json" and item.is_file():
                    keys.append(item.stem)

        return sorted(keys, reverse=True)

    def delete(self, key: str) -> None:
        """Delete an item by its key.

        Args:
            key: The identifier.
        """
        item_path = self._item_path(key)
        if item_path.exists():
            if self.use_subdirs:
                shutil.rmtree(item_path)
            else:
                item_path.unlink()

    def clear(self) -> None:
        """Delete all items."""
        if self.path.exists():
            shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[T]:
        """Load all items.

        Returns:
            List of deserialized data.
        """
        return [self.load(key) for key in self.list_keys()]


class RunStorage(FileStorage[dict[str, Any]]):
    """Persist simulation runs to disk for later analysis and reporting."""

    def __init__(self, path: Path | str = ".understudy/runs"):
        super().__init__(path, use_subdirs=True)

    def _generate_key(self, **kwargs: Any) -> str:
        trace: Trace = kwargs["trace"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = secrets.token_hex(3)
        return f"{trace.scene_id}_{timestamp}_{suffix}"

    def _serialize(self, key: str, **kwargs: Any) -> dict[str, Any]:
        trace: Trace = kwargs["trace"]
        scene: Scene = kwargs["scene"]
        judges = kwargs.get("judges")
        check_result = kwargs.get("check_result")
        tags = kwargs.get("tags")

        files: dict[str, Any] = {}

        files["trace.json"] = json.loads(trace.model_dump_json())
        files["scene.json"] = json.loads(scene.model_dump_json())

        if judges:
            judges_data = {}
            for name, result in judges.items():
                if hasattr(result, "model_dump"):
                    judges_data[name] = result.model_dump()
                elif hasattr(result, "__dict__"):
                    judges_data[name] = result.__dict__
                else:
                    judges_data[name] = result
            files["judges.json"] = judges_data

        if check_result:
            check_data = {
                "passed": check_result.passed,
                "checks": [
                    {"label": c.label, "passed": c.passed, "detail": c.detail}
                    for c in check_result.checks
                ],
            }
            files["check.json"] = check_data

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "run_id": key,
            "scene_id": trace.scene_id,
            "timestamp": timestamp,
            "passed": check_result.passed if check_result else None,
            "terminal_state": trace.terminal_state,
            "turn_count": trace.turn_count,
            "tools_called": trace.call_sequence(),
            "agents_invoked": trace.agents_invoked(),
            "tags": tags or {},
        }
        files["metadata.json"] = metadata

        return files

    def _deserialize(self, key: str, stored: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"run_id": key}

        if "trace.json" in stored:
            result["trace"] = Trace.model_validate(stored["trace.json"])

        if "scene.json" in stored:
            result["scene"] = Scene.model_validate(stored["scene.json"])

        if "judges.json" in stored:
            result["judges"] = stored["judges.json"]

        if "check.json" in stored:
            result["check"] = stored["check.json"]

        if "metadata.json" in stored:
            result["metadata"] = stored["metadata.json"]

        return result

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
        return self._save_internal(
            trace=trace,
            scene=scene,
            judges=judges,
            check_result=check_result,
            tags=tags,
        )

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


class TraceStorage(FileStorage[dict[str, Any]]):
    """Persist simulation traces to disk (without evaluation results).

    Used by simulate/simulate_batch for simulation-only workflows.
    """

    def __init__(self, path: Path | str = ".understudy/traces"):
        super().__init__(path, use_subdirs=False)

    def _generate_key(self, **kwargs: Any) -> str:
        trace: Trace = kwargs["trace"]
        sim_index = kwargs.get("sim_index", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{trace.scene_id}_{sim_index}_{timestamp}"

    def _serialize(self, key: str, **kwargs: Any) -> dict[str, Any]:
        trace: Trace = kwargs["trace"]
        scene: Scene = kwargs["scene"]
        sim_index = kwargs.get("sim_index", 0)
        tags = kwargs.get("tags")

        return {
            "trace": trace.model_dump(mode="json"),
            "scene": scene.model_dump(mode="json"),
            "metadata": {
                "trace_id": key,
                "scene_id": trace.scene_id,
                "sim_index": sim_index,
                "timestamp": datetime.now().isoformat(),
                "tags": tags or {},
            },
        }

    def _deserialize(self, key: str, stored: dict[str, Any]) -> dict[str, Any]:
        del key
        stored["trace"] = Trace.model_validate(stored["trace"])
        stored["scene"] = Scene.model_validate(stored["scene"])
        return stored

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
        return self._save_internal(trace=trace, scene=scene, sim_index=sim_index, tags=tags)

    def load_trace(self, trace_id: str) -> Trace:
        """Load just the trace object by its ID."""
        return self.load(trace_id)["trace"]

    def list_traces(self) -> list[str]:
        """List all trace IDs in storage."""
        return self.list_keys()


class EvaluationStorage(FileStorage[dict[str, Any]]):
    """Persist evaluation results to disk.

    Used by evaluate/evaluate_batch for storing evaluation results.
    """

    def __init__(self, path: Path | str = ".understudy/results"):
        super().__init__(path, use_subdirs=False)

    def _generate_key(self, **kwargs: Any) -> str:
        trace_id: str = kwargs["trace_id"]
        return f"{trace_id}_eval"

    def _serialize(self, key: str, **kwargs: Any) -> dict[str, Any]:
        del key
        trace_id: str = kwargs["trace_id"]
        check_result: CheckResult = kwargs["check_result"]
        judges = kwargs.get("judges")

        serialized: dict[str, Any] = {
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
            serialized["judges"] = judges_data

        return serialized

    def _deserialize(self, key: str, stored: dict[str, Any]) -> dict[str, Any]:
        del key
        return stored

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
        return self._save_internal(trace_id=trace_id, check_result=check_result, judges=judges)

    def list_results(self) -> list[str]:
        """List all result IDs in storage."""
        return self.list_keys()
