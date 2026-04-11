"""CLI: command-line interface for understudy."""

import importlib
import sys
from pathlib import Path
from typing import Any

import click

from .compare import compare_runs
from .reports import ReportGenerator
from .storage import RunStorage
from .validation import SceneValidationError


def import_object(import_path: str) -> Any:
    """Import a Python object from an import path.

    Args:
        import_path: Import path in the format "module:attribute" or "module.submodule:attribute".

    Returns:
        The imported object.

    Raises:
        click.ClickException: If the import path is invalid or the object cannot be found.
    """
    if ":" not in import_path:
        raise click.ClickException(
            f"Invalid import path '{import_path}'. Expected format: 'module:attribute'"
        )

    module_path, attr_name = import_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise click.ClickException(f"Cannot import module '{module_path}': {e}") from e

    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise click.ClickException(f"Module '{module_path}' has no attribute '{attr_name}'") from e


@click.group()
@click.version_option()
def main():
    """understudy - Test your AI agents with simulated users."""
    pass


@main.command()
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="report",
    help="Output directory for report",
)
@click.option(
    "--analyze-failures",
    is_flag=True,
    help="Use LLM to analyze why failed runs failed",
)
@click.option(
    "--analysis-model",
    default="gpt-4o",
    help="Model to use for failure analysis",
)
def report(runs: Path, output: Path, analyze_failures: bool, analysis_model: str):
    """Generate a static HTML report from saved runs."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        sys.exit(1)

    click.echo(f"Found {len(run_ids)} runs")

    if analyze_failures:
        failed = sum(1 for r in run_ids if not storage.load(r).get("metadata", {}).get("passed"))
        if failed:
            click.echo(f"Analyzing {failed} failed runs with {analysis_model}...")

    generator = ReportGenerator(
        storage,
        analyze_failures=analyze_failures,
        analysis_model=analysis_model,
    )
    generator.generate_static_report(output)

    click.echo(f"Report generated: {output}")


@main.command()
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8080,
    help="Port to serve on",
)
@click.option(
    "--host",
    "-h",
    type=str,
    default="127.0.0.1",
    help="Host to bind to",
)
def serve(runs: Path, port: int, host: str):
    """Start an interactive report browser."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        sys.exit(1)

    click.echo(f"Found {len(run_ids)} runs")

    generator = ReportGenerator(storage)
    generator.serve(port=port, host=host)


@main.command("list")
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
def list_runs(runs: Path):
    """List all saved runs."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        return

    click.echo(f"Found {len(run_ids)} runs:\n")

    for run_id in run_ids:
        data = storage.load(run_id)
        meta = data.get("metadata", {})
        status = "PASS" if meta.get("passed") else "FAIL"
        state = meta.get("terminal_state", "none")
        turns = meta.get("turn_count", 0)
        tags = meta.get("tags", {})
        tag_str = " ".join(f"[{k}={v}]" for k, v in tags.items()) if tags else ""
        line = f"  [{status}] {run_id} - {state} ({turns} turns)"
        if tag_str:
            line = f"{line} {tag_str}"
        click.echo(line)


@main.command()
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
def summary(runs: Path):
    """Show aggregate metrics for saved runs."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        return

    stats = storage.get_summary()

    click.echo("understudy Summary")
    click.echo("=" * 40)
    click.echo(f"Total Runs:    {stats['total_runs']}")
    click.echo(f"Pass Rate:     {stats['pass_rate'] * 100:.1f}%")
    click.echo(f"Avg Turns:     {stats['avg_turns']:.1f}")

    if stats["tool_usage"]:
        click.echo("\nTool Usage:")
        for tool, count in sorted(stats["tool_usage"].items(), key=lambda x: -x[1]):
            click.echo(f"  {tool}: {count}")

    if stats["terminal_states"]:
        click.echo("\nTerminal States:")
        for state, count in sorted(stats["terminal_states"].items(), key=lambda x: -x[1]):
            click.echo(f"  {state}: {count}")

    if stats["agents"]:
        click.echo("\nAgents:")
        for agent, count in sorted(stats["agents"].items(), key=lambda x: -x[1]):
            click.echo(f"  {agent}: {count}")


@main.command()
@click.argument("run_id")
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
def show(run_id: str, runs: Path):
    """Show details for a specific run."""
    storage = RunStorage(path=runs)

    try:
        data = storage.load(run_id)
    except FileNotFoundError:
        click.echo(f"Run not found: {run_id}")
        sys.exit(1)

    meta = data.get("metadata", {})
    trace = data.get("trace")
    check = data.get("check")

    click.echo(f"Run: {run_id}")
    click.echo("=" * 40)
    click.echo(f"Scene:          {meta.get('scene_id', 'unknown')}")
    click.echo(f"Status:         {'PASS' if meta.get('passed') else 'FAIL'}")
    click.echo(f"Terminal State: {meta.get('terminal_state', 'none')}")
    click.echo(f"Turns:          {meta.get('turn_count', 0)}")
    click.echo(f"Tools Called:   {', '.join(meta.get('tools_called', []))}")
    click.echo(f"Agents:         {', '.join(meta.get('agents_invoked', []))}")

    tags = meta.get("tags", {})
    if tags:
        click.echo(f"Tags:           {', '.join(f'{k}={v}' for k, v in tags.items())}")

    if check and check.get("checks"):
        click.echo("\nExpectation Checks:")
        for c in check["checks"]:
            icon = "+" if c["passed"] else "-"
            click.echo(f"  {icon} {c['label']}: {c['detail']}")

    if trace:
        click.echo("\nConversation:")
        click.echo("-" * 40)
        for turn in trace.turns:
            role = turn.agent_name or turn.role.upper()
            click.echo(f"[{role}]: {turn.content[:100]}{'...' if len(turn.content) > 100 else ''}")
            for call in turn.tool_calls:
                click.echo(f"  -> {call.tool_name}({call.arguments})")


@main.command()
@click.argument("run_id")
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def delete(run_id: str, runs: Path, yes: bool):
    """Delete a specific run."""
    storage = RunStorage(path=runs)

    try:
        storage.load(run_id)
    except FileNotFoundError:
        click.echo(f"Run not found: {run_id}")
        sys.exit(1)

    if not yes and not click.confirm(f"Delete run {run_id}?"):
        return

    storage.delete(run_id)
    click.echo(f"Deleted: {run_id}")


@main.command()
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def clear(runs: Path, yes: bool):
    """Delete all saved runs."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        return

    if not yes and not click.confirm(f"Delete all {len(run_ids)} runs?"):
        return

    storage.clear()
    click.echo(f"Cleared {len(run_ids)} runs")


@main.command()
@click.option(
    "--runs",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=".understudy/runs",
    help="Path to runs directory",
)
@click.option("--tag", "-t", required=True, help="Tag key to compare on")
@click.option("--before", "-A", required=True, help="Baseline tag value")
@click.option("--after", "-B", required=True, help="Candidate tag value")
@click.option("--before-label", help="Display label for baseline")
@click.option("--after-label", help="Display label for candidate")
@click.option(
    "--html",
    "-o",
    type=click.Path(path_type=Path),
    help="Output HTML report to file",
)
def compare(
    runs: Path,
    tag: str,
    before: str,
    after: str,
    before_label: str | None,
    after_label: str | None,
    html: Path | None,
):
    """Compare runs between two tag values."""
    storage = RunStorage(path=runs)

    if html:
        generator = ReportGenerator(storage)
        try:
            content = generator.generate_comparison_report(
                tag=tag,
                before_value=before,
                after_value=after,
                before_label=before_label,
                after_label=after_label,
            )
        except ValueError as e:
            click.echo(f"Error: {e}")
            sys.exit(1)

        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(content)
        click.echo(f"Comparison report generated: {html}")
        return

    try:
        result = compare_runs(
            storage,
            tag=tag,
            before_value=before,
            after_value=after,
            before_label=before_label,
            after_label=after_label,
        )
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    click.echo(f"Comparison: {result.before_label} vs {result.after_label}")
    click.echo(f"Tag: {result.tag}")
    click.echo("=" * 50)

    click.echo(f"\n{'Metric':<20} {result.before_label:>12} {result.after_label:>12} {'Delta':>12}")
    click.echo("-" * 56)

    runs_delta = result.after_runs - result.before_runs
    click.echo(f"{'Runs':<20} {result.before_runs:>12} {result.after_runs:>12} {runs_delta:>+12}")

    b_pr = result.before_pass_rate * 100
    a_pr = result.after_pass_rate * 100
    d_pr = result.pass_rate_delta * 100
    click.echo(f"{'Pass Rate':<20} {b_pr:>11.1f}% {a_pr:>11.1f}% {d_pr:>+11.1f}%")

    b_turns = result.before_avg_turns
    a_turns = result.after_avg_turns
    d_turns = result.avg_turns_delta
    click.echo(f"{'Avg Turns':<20} {b_turns:>12.1f} {a_turns:>12.1f} {d_turns:>+12.1f}")

    click.echo("\nTerminal States:")
    all_states = set(result.terminal_states_before.keys()) | set(
        result.terminal_states_after.keys()
    )
    for state in sorted(all_states):
        b = result.terminal_states_before.get(state, 0)
        a = result.terminal_states_after.get(state, 0)
        click.echo(f"  {state:<18} {b:>12} {a:>12} {a - b:>+12}")

    click.echo("\nTool Usage:")
    all_tools = set(result.tool_usage_before.keys()) | set(result.tool_usage_after.keys())
    for tool in sorted(all_tools):
        b = result.tool_usage_before.get(tool, 0)
        a = result.tool_usage_after.get(tool, 0)
        click.echo(f"  {tool:<18} {b:>12} {a:>12} {a - b:>+12}")

    if result.per_scene:
        click.echo("\nPer-Scene Breakdown:")
        hdr = f"  {'Scene':<30} {result.before_label:>12} {result.after_label:>12} {'Delta':>12}"
        click.echo(hdr)
        click.echo("  " + "-" * 66)
        for sc in result.per_scene:
            b_str = f"{sc.before_passed}/{sc.before_total}" if sc.before_total else "-"
            a_str = f"{sc.after_passed}/{sc.after_total}" if sc.after_total else "-"
            d_pct = sc.pass_rate_delta * 100
            click.echo(f"  {sc.scene_id:<30} {b_str:>12} {a_str:>12} {d_pct:>+11.0f}%")


@main.command("serve-api")
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to serve on (default: 8000)",
)
@click.option(
    "--host",
    "-h",
    type=str,
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--simulator-model",
    default="gpt-4o",
    help="Model for user simulator (default: gpt-4o)",
)
def serve_api(port: int, host: str, simulator_model: str):
    """Start the HTTP simulator API server for browser/UI testing."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn not installed. Install with: pip install understudy[server]")
        sys.exit(1)

    try:
        from .server import get_app
    except ImportError as e:
        click.echo("Error: FastAPI not installed. Install with: pip install understudy[server]")
        click.echo(f"Details: {e}")
        sys.exit(1)

    app = get_app(model=simulator_model)

    click.echo("Starting understudy HTTP simulator API")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Simulator model: {simulator_model}")
    click.echo("\nAPI endpoints:")
    click.echo(f"  POST   http://{host}:{port}/sessions           - Create session")
    click.echo(f"  POST   http://{host}:{port}/sessions/{{id}}/turn  - Process turn")
    click.echo(f"  POST   http://{host}:{port}/sessions/{{id}}/evaluate - Evaluate")
    click.echo(f"  GET    http://{host}:{port}/sessions/{{id}}/trace - Get trace")
    click.echo(f"  DELETE http://{host}:{port}/sessions/{{id}}       - Delete session")
    click.echo("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command("simulate")
@click.option(
    "--app",
    required=True,
    help="Python import path to AgentApp (e.g., mymodule:my_app)",
)
@click.option(
    "--scenes",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to scene file (.yaml/.json) or directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=".understudy/traces",
    help="Output directory for trace files",
)
@click.option(
    "--simulator-model",
    default="gpt-4o",
    help="Model for user simulator (default: gpt-4o)",
)
@click.option(
    "--n-sims",
    type=int,
    default=1,
    help="Number of simulations per scene (default: 1)",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of scenes to run in parallel (default: 1)",
)
@click.option(
    "--mocks",
    "mocks_path",
    default=None,
    help="Python import path to mocks function (returns MockToolkit)",
)
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Add tag (repeatable, format: key=value)",
)
def simulate_command(
    app: str,
    scenes: Path,
    output: Path,
    simulator_model: str,
    n_sims: int,
    parallel: int,
    mocks_path: str | None,
    tags: tuple[str, ...],
):
    """Run simulations only (no evaluation)."""
    from .runner import simulate_batch

    sys.path.insert(0, str(Path.cwd()))

    agent_app = import_object(app)

    mocks = None
    if mocks_path:
        mocks_fn = import_object(mocks_path)
        mocks = mocks_fn()

    tags_dict: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            raise click.ClickException(f"Invalid tag format '{tag}'. Expected 'key=value'")
        key, value = tag.split("=", 1)
        tags_dict[key] = value

    click.echo(f"Running simulations with model: {simulator_model}")
    if mocks:
        click.echo(f"Using mocks: {mocks.available_tools}")

    traces = simulate_batch(
        app=agent_app,
        scenes=scenes,
        simulator_model=simulator_model,
        n_sims=n_sims,
        parallel=parallel,
        mocks=mocks,
        output=output,
        tags=tags_dict if tags_dict else None,
    )

    scene_count = len(set(t.scene_id for t in traces))
    click.echo(f"\nSimulated {scene_count} scenes x {n_sims} runs = {len(traces)} traces")
    click.echo(f"Traces saved to: {output}")


@main.command("evaluate")
@click.option(
    "--traces",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trace file or directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=".understudy/results",
    help="Output directory for evaluation results",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for LLM judge evaluation",
)
@click.option(
    "--metrics",
    default=None,
    help="Metrics to compute (comma-separated, default: from scene)",
)
@click.option(
    "--junit",
    type=click.Path(path_type=Path),
    default=None,
    help="Export JUnit XML to path",
)
def evaluate_command(
    traces: Path,
    output: Path,
    judge_model: str | None,
    metrics: str | None,
    junit: Path | None,
):
    """Evaluate existing traces."""
    from .check import evaluate_batch

    metrics_list = [m.strip() for m in metrics.split(",")] if metrics else None

    click.echo(f"Evaluating traces from: {traces}")
    if judge_model:
        click.echo(f"Using judge model: {judge_model}")

    results = evaluate_batch(
        traces=traces,
        output=output,
        judge_model=judge_model,
        metrics=metrics_list,
    )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    click.echo(f"\nEvaluation Results: {passed}/{len(results)} passed")
    if failed > 0:
        click.echo("\nFailed:")
        for r in results:
            if not r.passed:
                if r.error:
                    click.echo(f"  - {r.trace_id}: {r.error}")
                else:
                    for c in r.check_result.failed_checks:
                        click.echo(f"  - {r.trace_id}: {c.label}: {c.detail}")

    click.echo(f"\nResults saved to: {output}")

    if junit:
        from .suite import SceneResult, SuiteResults
        from .trace import Trace

        suite_results = SuiteResults(
            results=[
                SceneResult(
                    scene_id=r.trace_id,
                    trace=Trace(scene_id=r.trace_id),
                    check_result=r.check_result,
                    error=r.error,
                )
                for r in results
            ]
        )
        suite_results.to_junit_xml(junit)
        click.echo(f"JUnit XML exported to: {junit}")

    if failed > 0:
        sys.exit(1)


@main.command("run")
@click.option(
    "--app",
    required=True,
    help="Python import path to AgentApp (e.g., mymodule:my_app)",
)
@click.option(
    "--scene",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to scene file (.yaml/.json) or directory",
)
@click.option(
    "--simulator-model",
    default="gpt-4o",
    help="Model for user simulator (default: gpt-4o)",
)
@click.option(
    "--judge-model",
    default=None,
    help="Model for LLM judge evaluation (if set, runs judges)",
)
@click.option(
    "--rubric",
    default="all",
    help="Rubrics to evaluate: 'all' or comma-separated list",
)
@click.option(
    "--mocks",
    "mocks_path",
    default=None,
    help="Python import path to mocks function (returns MockToolkit)",
)
@click.option(
    "--runs",
    "-r",
    type=click.Path(path_type=Path),
    default=".understudy/runs",
    help="Storage path for results",
)
@click.option(
    "--parallel",
    type=int,
    default=1,
    help="Number of scenes to run in parallel (default: 1)",
)
@click.option(
    "--n-sims",
    type=int,
    default=1,
    help="Number of simulations per scene (default: 1)",
)
@click.option(
    "--tag",
    "-t",
    "tags",
    multiple=True,
    help="Add tag (repeatable, format: key=value)",
)
@click.option(
    "--junit",
    type=click.Path(path_type=Path),
    default=None,
    help="Export JUnit XML to path",
)
def run_command(
    app: str,
    scene: Path,
    simulator_model: str,
    judge_model: str | None,
    rubric: str,
    mocks_path: str | None,
    runs: Path,
    parallel: int,
    n_sims: int,
    tags: tuple[str, ...],
    junit: Path | None,
):
    """Run simulations against an agent app (simulate + evaluate)."""
    from .judges import Judge
    from .models import Scene as SceneModel
    from .prompts import rubrics as rubric_module
    from .storage import RunStorage
    from .suite import Suite

    sys.path.insert(0, str(Path.cwd()))

    agent_app = import_object(app)

    mocks = None
    if mocks_path:
        mocks_fn = import_object(mocks_path)
        mocks = mocks_fn()

    tags_dict: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            raise click.ClickException(f"Invalid tag format '{tag}'. Expected 'key=value'")
        key, value = tag.split("=", 1)
        tags_dict[key] = value

    try:
        if scene.is_dir():
            suite = Suite.from_directory(scene)
            click.echo(f"Loaded {len(suite.scenes)} scenes from {scene}")
        else:
            scene_obj = SceneModel.from_file(scene)
            suite = Suite([scene_obj])
            click.echo(f"Loaded scene: {scene_obj.id}")
    except SceneValidationError as e:
        raise click.ClickException(e.message) from e

    storage = RunStorage(path=runs)

    click.echo(f"Running with simulator model: {simulator_model}")
    if n_sims > 1:
        click.echo(f"Simulations per scene: {n_sims}")
    if mocks:
        click.echo(f"Using mocks: {mocks.available_tools}")

    results = suite.run(
        app=agent_app,
        parallel=parallel,
        storage=storage,
        tags=tags_dict if tags_dict else None,
        n_sims=n_sims,
        mocks=mocks,
        simulator_model=simulator_model,
    )

    click.echo("\n" + results.summary())

    if judge_model:
        click.echo(f"\nRunning judge evaluations with model: {judge_model}")

        if rubric == "all":
            rubric_names = [
                name for name in dir(rubric_module) if name.isupper() and not name.startswith("_")
            ]
        else:
            rubric_names = [r.strip() for r in rubric.split(",")]

        for scene_result in results.results:
            click.echo(f"\nJudge results for {scene_result.scene_id}:")
            judge_results = {}
            for rubric_name in rubric_names:
                rubric_text = getattr(rubric_module, rubric_name, None)
                if rubric_text is None:
                    click.echo(f"  Warning: Unknown rubric '{rubric_name}'")
                    continue

                judge = Judge(rubric=rubric_text, model=judge_model)
                result = judge.evaluate(scene_result.trace)
                judge_results[rubric_name] = result
                status = "PASS" if result.score == 1 else "FAIL"
                click.echo(f"  [{status}] {rubric_name} (agreement: {result.agreement_rate:.0%})")

    if junit:
        results.to_junit_xml(junit)
        click.echo(f"\nJUnit XML exported to: {junit}")

    if results.all_passed:
        click.echo("\nAll scenes passed!")
        sys.exit(0)
    else:
        click.echo("\nSome scenes failed.")
        sys.exit(1)


@main.command("init")
@click.argument("path", type=click.Path(path_type=Path), default=".")
@click.option(
    "--adapter",
    type=click.Choice(["generic", "adk", "langgraph", "http"]),
    default=None,
    help="Agent adapter type for generated examples",
)
def init_command(path: Path, adapter: str | None):
    """Initialize a new understudy test project.

    Creates a project structure with example scenes, test files, and configuration.

    Examples:

        understudy init                  # Initialize in current directory

        understudy init my-agent-tests   # Create new directory

        understudy init --adapter adk    # With ADK adapter examples
    """
    import shutil

    templates_dir = Path(__file__).parent / "templates" / "init"

    if not templates_dir.exists():
        raise click.ClickException(f"Template directory not found: {templates_dir}")

    path = Path(path)
    if path != Path("."):
        if path.exists() and any(path.iterdir()):
            raise click.ClickException(f"Directory '{path}' already exists and is not empty")
        path.mkdir(parents=True, exist_ok=True)

    scenes_dir = path / "scenes"
    scenes_dir.mkdir(exist_ok=True)

    example_scene = templates_dir / "example_scene.yaml"
    if example_scene.exists():
        shutil.copy(example_scene, scenes_dir / "example_scene.yaml")
        click.echo(f"  Created {scenes_dir / 'example_scene.yaml'}")

    conftest_template = templates_dir / "conftest.py.template"
    if conftest_template.exists():
        content = conftest_template.read_text()
        if adapter == "adk":
            content = _customize_conftest_adk(content)
        elif adapter == "langgraph":
            content = _customize_conftest_langgraph(content)
        elif adapter == "http":
            content = _customize_conftest_http(content)
        (path / "conftest.py").write_text(content)
        click.echo(f"  Created {path / 'conftest.py'}")

    test_template = templates_dir / "test_agent.py.template"
    if test_template.exists():
        shutil.copy(test_template, path / "test_agent.py")
        click.echo(f"  Created {path / 'test_agent.py'}")

    config_template = templates_dir / "understudy.yaml.template"
    if config_template.exists():
        shutil.copy(config_template, path / ".understudy.yaml")
        click.echo(f"  Created {path / '.understudy.yaml'}")

    click.echo(f"\nProject initialized in {path.absolute()}")
    click.echo("\nNext steps:")
    click.echo("  1. Configure your agent app in conftest.py")
    click.echo("  2. Customize the example scene in scenes/example_scene.yaml")
    click.echo("  3. Run tests: pytest test_agent.py")
    click.echo("\nOr use the CLI directly:")
    click.echo("  understudy run --app mymodule:app --scene scenes/")


def _get_mocks_section(content: str) -> str:
    """Extract the mocks fixture section from conftest template."""
    marker = "@pytest.fixture\ndef mocks():"
    if marker in content:
        return marker + content.split(marker)[1]
    return ""


def _customize_conftest_adk(content: str) -> str:
    """Customize conftest.py for ADK adapter."""
    mocks_section = _get_mocks_section(content)
    return f'''"""Pytest configuration for understudy tests."""

import pytest

from understudy import MockToolkit
from understudy.adk import ADKApp

# TODO: Import your ADK agent
# from myagent import root_agent


@pytest.fixture
def app():
    """Return your ADK agent wrapped in ADKApp.

    Example:
        from myagent import root_agent
        return ADKApp(root_agent)
    """
    # TODO: Return your ADK app
    # return ADKApp(root_agent)
    raise NotImplementedError("Configure your ADK agent in conftest.py")


{mocks_section}
'''


def _customize_conftest_langgraph(content: str) -> str:
    """Customize conftest.py for LangGraph adapter."""
    mocks_section = _get_mocks_section(content)
    return f'''"""Pytest configuration for understudy tests."""

import pytest

from understudy import MockToolkit
from understudy.langgraph import LangGraphApp

# TODO: Import your LangGraph graph
# from myagent import graph


@pytest.fixture
def app():
    """Return your LangGraph graph wrapped in LangGraphApp.

    Example:
        from myagent import graph
        return LangGraphApp(graph)
    """
    # TODO: Return your LangGraph app
    # return LangGraphApp(graph)
    raise NotImplementedError("Configure your LangGraph graph in conftest.py")


{mocks_section}
'''


def _customize_conftest_http(content: str) -> str:
    """Customize conftest.py for HTTP adapter."""
    mocks_section = _get_mocks_section(content)
    return f'''"""Pytest configuration for understudy tests."""

import pytest

from understudy import MockToolkit
from understudy.http import HTTPApp

# TODO: Configure your HTTP endpoint
# AGENT_URL = "http://localhost:8000/chat"


@pytest.fixture
def app():
    """Return your HTTP agent wrapped in HTTPApp.

    Example:
        return HTTPApp(url="http://localhost:8000/chat")
    """
    # TODO: Return your HTTP app
    # return HTTPApp(url="http://localhost:8000/chat")
    raise NotImplementedError("Configure your HTTP endpoint in conftest.py")


{mocks_section}
'''


@main.command("diff")
@click.argument("trace1", type=click.Path(exists=True, path_type=Path))
@click.argument("trace2", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--html",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output HTML diff report to file",
)
def diff_command(trace1: Path, trace2: Path, html: Path | None):
    """Compare two trace files and show differences.

    Useful for detecting regressions when comparing agent versions.

    Examples:

        understudy diff trace_v1.json trace_v2.json

        understudy diff run1/trace.json run2/trace.json --html diff.html
    """
    from .diff import diff_traces
    from .replay import load_trace

    t1 = load_trace(trace1)
    t2 = load_trace(trace2)

    result = diff_traces(t1, t2)

    if html:
        content = _generate_diff_html(result, trace1, trace2)
        html.parent.mkdir(parents=True, exist_ok=True)
        html.write_text(content)
        click.echo(f"Diff report generated: {html}")
    else:
        click.echo(result.summary())

        if result.regression_warnings:
            sys.exit(1)


@main.command("replay")
@click.argument("trace", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--app",
    required=True,
    help="Python import path to AgentApp (e.g., mymodule:my_app)",
)
@click.option(
    "--mocks",
    "mocks_path",
    default=None,
    help="Python import path to mocks function (returns MockToolkit)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for new trace file",
)
@click.option(
    "--diff",
    "show_diff",
    is_flag=True,
    help="Show diff between original and replay traces",
)
def replay_command(
    trace: Path,
    app: str,
    mocks_path: str | None,
    output: Path | None,
    show_diff: bool,
):
    """Replay a recorded trace against a new agent version.

    This sends the original user messages to the new agent and compares behavior.

    Examples:

        understudy replay trace.json --app mymodule:new_app

        understudy replay trace.json --app mymodule:app --diff

        understudy replay trace.json --app mymodule:app --output new_trace.json
    """
    from .diff import diff_traces
    from .replay import load_trace, replay

    sys.path.insert(0, str(Path.cwd()))

    agent_app = import_object(app)

    mocks = None
    if mocks_path:
        mocks_fn = import_object(mocks_path)
        mocks = mocks_fn()

    original = load_trace(trace)

    click.echo(f"Replaying trace: {original.scene_id}")
    click.echo(f"Original turns: {original.turn_count}")

    result = replay(original, agent_app, mocks=mocks)

    click.echo("\nReplay Results:")
    click.echo(f"  Match rate: {result.match_rate:.1%}")
    click.echo(f"  Matched: {result.matched_responses}/{result.total_turns} turns")

    if result.diverged_at_turn is not None:
        click.echo(f"  Diverged at turn: {result.diverged_at_turn}")

    if result.errors:
        click.echo("\nErrors:")
        for err in result.errors:
            click.echo(f"  - {err}")

    if show_diff:
        click.echo("\n" + "=" * 50)
        diff_result = diff_traces(original, result.new_trace)
        click.echo(diff_result.summary())

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.new_trace.model_dump_json(indent=2))
        click.echo(f"\nNew trace saved to: {output}")

    if not result.fully_matched:
        sys.exit(1)


def _generate_diff_html(diff, trace1_path: Path, trace2_path: Path) -> str:
    """Generate an HTML diff report."""
    warnings_html = ""
    if diff.regression_warnings:
        warnings_html = "".join(
            f'<div class="warning">! {w}</div>' for w in diff.regression_warnings
        )

    added_html = "<p>None</p>"
    if diff.added_tools:
        items = "".join(f"<li class='added'>+ {t}</li>" for t in diff.added_tools)
        added_html = f"<ul class='tool-list'>{items}</ul>"

    removed_html = "<p>None</p>"
    if diff.removed_tools:
        items = "".join(f"<li class='removed'>- {t}</li>" for t in diff.removed_tools)
        removed_html = f"<ul class='tool-list'>{items}</ul>"

    changed_html = "<p>None</p>"
    if diff.changed_calls:
        rows = []
        for c in diff.changed_calls:
            changes = ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in c.arg_changes.items())
            rows.append(f"<tr><td>{c.tool_name}</td><td>{changes}</td></tr>")
        changed_html = f"<table><tr><th>Tool</th><th>Changes</th></tr>{''.join(rows)}</table>"

    terminal_html = ""
    if diff.terminal_state_changed:
        terminal_html = (
            f"<div class='section'><h2>Terminal State</h2>"
            f"<p>{diff.trace1_terminal} -> {diff.trace2_terminal}</p></div>"
        )

    added_count = f"+{len(diff.added_tools)}" if diff.added_tools else "0"
    removed_count = f"-{len(diff.removed_tools)}" if diff.removed_tools else "0"
    changed_count = f"~{len(diff.changed_calls)}" if diff.changed_calls else "0"

    css = """
        body { font-family: system-ui, sans-serif; max-width: 1000px; margin: auto; padding: 20px; }
        h1 { color: #333; }
        .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
        .added { color: #22863a; background: #dcffe4; padding: 2px 5px; }
        .removed { color: #cb2431; background: #ffeef0; padding: 2px 5px; }
        .changed { color: #b08800; background: #fff5b1; padding: 2px 5px; }
        .warning { background: #fffbdd; border-left: 4px solid #b08800; padding: 10px; }
        .section { margin: 20px 0; }
        .tool-list { list-style: none; padding: 0; }
        .tool-list li { padding: 5px 10px; margin: 2px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
    """

    changes_line = f"{added_count} added, {removed_count} removed, {changed_count} changed"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Trace Diff: {diff.trace1_id} vs {diff.trace2_id}</title>
    <style>{css}</style>
</head>
<body>
    <h1>Trace Diff</h1>
    <div class="summary">
        <strong>Before:</strong> {trace1_path.name} ({diff.trace1_id})<br>
        <strong>After:</strong> {trace2_path.name} ({diff.trace2_id})<br>
        <strong>Changes:</strong> {changes_line}
    </div>
    {warnings_html}
    <div class="section"><h2>Added Tools</h2>{added_html}</div>
    <div class="section"><h2>Removed Tools</h2>{removed_html}</div>
    <div class="section"><h2>Changed Calls</h2>{changed_html}</div>
    {terminal_html}
</body>
</html>
"""


@main.command("run-agentic")
@click.option(
    "--app",
    required=True,
    help="Python import path to AgenticApp (e.g., mymodule:my_app)",
)
@click.option(
    "--scene",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to agentic scene file (.yaml/.json)",
)
@click.option(
    "--mocks",
    "mocks_path",
    default=None,
    help="Python import path to mocks function (returns MockToolkit)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for trace file",
)
def run_agentic_command(
    app: str,
    scene: Path,
    mocks_path: str | None,
    output: Path | None,
):
    """Run an agentic scene against an autonomous agent."""
    from .agentic import AgenticScene, check_agentic, run_agentic

    sys.path.insert(0, str(Path.cwd()))

    agent_app = import_object(app)

    mocks = None
    if mocks_path:
        mocks_fn = import_object(mocks_path)
        mocks = mocks_fn()

    try:
        scene_obj = AgenticScene.from_file(scene)
        click.echo(f"Loaded agentic scene: {scene_obj.id}")
    except SceneValidationError as e:
        raise click.ClickException(e.message) from e

    click.echo(f"Task: {scene_obj.task.description}")
    click.echo(f"Goal: {scene_obj.task.goal}")
    if mocks:
        click.echo(f"Using mocks: {mocks.available_tools}")

    trace = run_agentic(agent_app, scene_obj, mocks=mocks)

    click.echo(f"\nOutcome: {trace.outcome}")
    click.echo(f"Steps: {trace.total_steps}")
    click.echo(f"Tokens: {trace.total_tokens}")

    result = check_agentic(trace, scene_obj.expectations)
    click.echo(f"\nChecks: {result}")
    click.echo(result.summary())

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(trace.model_dump_json(indent=2))
        click.echo(f"\nTrace saved to: {output}")

    if result.passed:
        click.echo("\nAll checks passed!")
        sys.exit(0)
    else:
        click.echo("\nSome checks failed.")
        sys.exit(1)


@main.command("evaluate-agentic")
@click.option(
    "--trace",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to agentic trace file (.json)",
)
@click.option(
    "--expectations",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to expectations file (.yaml/.json)",
)
@click.option(
    "--goal-predicate",
    default=None,
    help="Goal predicate expression (e.g., \"outcome == 'success'\")",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum allowed steps",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Maximum allowed tokens",
)
@click.option(
    "--required-actions",
    default=None,
    help="Required actions (comma-separated)",
)
@click.option(
    "--forbidden-actions",
    default=None,
    help="Forbidden actions (comma-separated, supports wildcards)",
)
def evaluate_agentic_command(
    trace: Path,
    expectations: Path | None,
    goal_predicate: str | None,
    max_steps: int | None,
    max_tokens: int | None,
    required_actions: str | None,
    forbidden_actions: str | None,
):
    """Evaluate an existing agentic trace."""
    import json

    import yaml

    from .agentic import AgenticExpectations, AgenticTrace, check_agentic
    from .agentic.metrics import compute_all_metrics

    trace_obj = AgenticTrace.from_file(trace)
    click.echo(f"Loaded trace: {trace_obj.scene_id}")
    click.echo(f"Outcome: {trace_obj.outcome}")
    click.echo(f"Steps: {trace_obj.total_steps}, Tokens: {trace_obj.total_tokens}")

    if expectations:
        with open(expectations) as f:
            if expectations.suffix in (".yaml", ".yml"):
                exp_data = yaml.safe_load(f)
            else:
                exp_data = json.load(f)
        exp_obj = AgenticExpectations(**exp_data)
    else:
        exp_obj = AgenticExpectations(
            goal_predicate=goal_predicate,
            max_steps=max_steps,
            max_tokens=max_tokens,
            required_actions=(
                [a.strip() for a in required_actions.split(",")]
                if required_actions
                else []
            ),
            forbidden_actions=(
                [a.strip() for a in forbidden_actions.split(",")]
                if forbidden_actions
                else []
            ),
        )

    result = check_agentic(trace_obj, exp_obj)
    click.echo(f"\nChecks: {result}")
    click.echo(result.summary())

    metrics = compute_all_metrics(trace_obj, exp_obj)
    click.echo("\nMetrics:")
    for name, metric in metrics.items():
        click.echo(f"  {name}: {metric.value}")

    if result.passed:
        click.echo("\nAll checks passed!")
        sys.exit(0)
    else:
        click.echo("\nSome checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
