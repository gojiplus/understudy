"""CLI: command-line interface for understudy."""

import sys
from pathlib import Path

import click

from .reports import ReportGenerator
from .storage import RunStorage


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
    default="report.html",
    help="Output HTML file path",
)
def report(runs: Path, output: Path):
    """Generate a static HTML report from saved runs."""
    storage = RunStorage(path=runs)

    run_ids = storage.list_runs()
    if not run_ids:
        click.echo(f"No runs found in {runs}")
        sys.exit(1)

    click.echo(f"Found {len(run_ids)} runs")

    generator = ReportGenerator(storage)
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
        click.echo(f"  [{status}] {run_id} - {state} ({turns} turns)")


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


if __name__ == "__main__":
    main()
