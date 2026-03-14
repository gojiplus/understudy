"""CLI: command-line interface for understudy."""

import sys
from pathlib import Path

import click

from .compare import compare_runs
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


if __name__ == "__main__":
    main()
