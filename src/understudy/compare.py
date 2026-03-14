"""Compare runs by tag values."""

from dataclasses import dataclass
from typing import Any

from .storage import RunStorage


@dataclass
class SceneComparison:
    """Per-scene comparison stats."""

    scene_id: str
    before_passed: int
    before_total: int
    after_passed: int
    after_total: int

    @property
    def before_pass_rate(self) -> float:
        return self.before_passed / self.before_total if self.before_total else 0.0

    @property
    def after_pass_rate(self) -> float:
        return self.after_passed / self.after_total if self.after_total else 0.0

    @property
    def pass_rate_delta(self) -> float:
        return self.after_pass_rate - self.before_pass_rate


@dataclass
class ComparisonResult:
    """Result of comparing two groups of runs."""

    tag: str
    before_value: str
    after_value: str
    before_label: str
    after_label: str
    before_runs: int
    after_runs: int
    before_pass_rate: float
    after_pass_rate: float
    pass_rate_delta: float
    before_avg_turns: float
    after_avg_turns: float
    avg_turns_delta: float
    tool_usage_before: dict[str, int]
    tool_usage_after: dict[str, int]
    terminal_states_before: dict[str, int]
    terminal_states_after: dict[str, int]
    per_scene: list[SceneComparison]


def compare_runs(
    storage: RunStorage,
    tag: str,
    before_value: str,
    after_value: str,
    before_label: str | None = None,
    after_label: str | None = None,
) -> ComparisonResult:
    """Compare runs grouped by tag values.

    Args:
        storage: RunStorage instance.
        tag: Tag key to filter on.
        before_value: Tag value for baseline group.
        after_value: Tag value for candidate group.
        before_label: Display label for baseline (defaults to before_value).
        after_label: Display label for candidate (defaults to after_value).

    Returns:
        ComparisonResult with metrics for both groups and deltas.

    Raises:
        ValueError: If either group has no matching runs.
    """
    all_runs = storage.load_all()

    before_runs = _filter_by_tag(all_runs, tag, before_value)
    after_runs = _filter_by_tag(all_runs, tag, after_value)

    if not before_runs:
        raise ValueError(f"No runs found with {tag}={before_value}")
    if not after_runs:
        raise ValueError(f"No runs found with {tag}={after_value}")

    before_stats = _compute_stats(before_runs)
    after_stats = _compute_stats(after_runs)
    per_scene = _compute_per_scene(before_runs, after_runs)

    return ComparisonResult(
        tag=tag,
        before_value=before_value,
        after_value=after_value,
        before_label=before_label or before_value,
        after_label=after_label or after_value,
        before_runs=len(before_runs),
        after_runs=len(after_runs),
        before_pass_rate=before_stats["pass_rate"],
        after_pass_rate=after_stats["pass_rate"],
        pass_rate_delta=after_stats["pass_rate"] - before_stats["pass_rate"],
        before_avg_turns=before_stats["avg_turns"],
        after_avg_turns=after_stats["avg_turns"],
        avg_turns_delta=after_stats["avg_turns"] - before_stats["avg_turns"],
        tool_usage_before=before_stats["tool_usage"],
        tool_usage_after=after_stats["tool_usage"],
        terminal_states_before=before_stats["terminal_states"],
        terminal_states_after=after_stats["terminal_states"],
        per_scene=per_scene,
    )


def _filter_by_tag(runs: list[dict[str, Any]], tag: str, value: str) -> list[dict[str, Any]]:
    """Filter runs by tag key/value."""
    return [r for r in runs if r.get("metadata", {}).get("tags", {}).get(tag) == value]


def _compute_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics for a group of runs."""
    if not runs:
        return {
            "pass_rate": 0.0,
            "avg_turns": 0.0,
            "tool_usage": {},
            "terminal_states": {},
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

    return {
        "pass_rate": passed / len(runs),
        "avg_turns": total_turns / len(runs),
        "tool_usage": tool_counts,
        "terminal_states": terminal_counts,
    }


def _compute_per_scene(
    before_runs: list[dict[str, Any]], after_runs: list[dict[str, Any]]
) -> list[SceneComparison]:
    """Compute per-scene pass rates for both groups."""
    before_by_scene: dict[str, dict[str, int]] = {}
    after_by_scene: dict[str, dict[str, int]] = {}

    for r in before_runs:
        scene_id = r.get("metadata", {}).get("scene_id", "unknown")
        if scene_id not in before_by_scene:
            before_by_scene[scene_id] = {"passed": 0, "total": 0}
        before_by_scene[scene_id]["total"] += 1
        if r.get("metadata", {}).get("passed"):
            before_by_scene[scene_id]["passed"] += 1

    for r in after_runs:
        scene_id = r.get("metadata", {}).get("scene_id", "unknown")
        if scene_id not in after_by_scene:
            after_by_scene[scene_id] = {"passed": 0, "total": 0}
        after_by_scene[scene_id]["total"] += 1
        if r.get("metadata", {}).get("passed"):
            after_by_scene[scene_id]["passed"] += 1

    all_scenes = sorted(set(before_by_scene.keys()) | set(after_by_scene.keys()))

    return [
        SceneComparison(
            scene_id=scene_id,
            before_passed=before_by_scene.get(scene_id, {}).get("passed", 0),
            before_total=before_by_scene.get(scene_id, {}).get("total", 0),
            after_passed=after_by_scene.get(scene_id, {}).get("passed", 0),
            after_total=after_by_scene.get(scene_id, {}).get("total", 0),
        )
        for scene_id in all_scenes
    ]
