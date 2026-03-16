"""Reports: generate HTML reports from saved runs."""

from pathlib import Path
from typing import Any

from .storage import RunStorage


class ReportGenerator:
    """Generate HTML reports from saved simulation runs."""

    def __init__(self, storage: RunStorage, static_mode: bool = False):
        """
        Args:
            storage: RunStorage instance containing saved runs.
            static_mode: If True, generate relative links for static HTML files.
        """
        self.storage = storage
        self.static_mode = static_mode
        self._env = None

    def _get_env(self):
        """Lazy-load Jinja2 environment."""
        if self._env is None:
            try:
                from jinja2 import Environment, PackageLoader
            except ImportError as e:
                raise ImportError(
                    "jinja2 package required. Install with: pip install understudy[reports]"
                ) from e

            self._env = Environment(
                loader=PackageLoader("understudy", "templates"),
                autoescape=True,
            )
            self._env.globals["static_mode"] = self.static_mode
        return self._env

    def generate_run_report(self, run_id: str) -> str:
        """Generate detailed HTML for a single run.

        Args:
            run_id: The run identifier.

        Returns:
            HTML content as a string.
        """
        env = self._get_env()
        data = self.storage.load(run_id)

        template = env.get_template("run_detail.html")
        return template.render(
            run_id=run_id,
            trace=data.get("trace"),
            scene=data.get("scene"),
            judges=data.get("judges"),
            check=data.get("check"),
            metadata=data.get("metadata", {}),
            base_path="../" if self.static_mode else "",
        )

    def generate_index(self) -> str:
        """Generate summary HTML listing all runs.

        Returns:
            HTML content as a string.
        """
        env = self._get_env()
        runs = []
        failed_runs = []

        for run_id in self.storage.list_runs():
            data = self.storage.load(run_id)
            meta = data.get("metadata", {})
            check_data = data.get("check", {})

            failed_checks = []
            for c in check_data.get("checks", []):
                if not c.get("passed"):
                    failed_checks.append(c.get("label", "unknown"))

            run_info = {
                "run_id": run_id,
                "scene_id": meta.get("scene_id", run_id),
                "passed": meta.get("passed"),
                "terminal_state": meta.get("terminal_state"),
                "turn_count": meta.get("turn_count", 0),
                "tools_called": meta.get("tools_called", []),
                "tags": meta.get("tags", {}),
                "timestamp": meta.get("timestamp", ""),
                "failed_checks": failed_checks,
            }
            runs.append(run_info)

            if not meta.get("passed"):
                failed_runs.append(run_info)

        summary = self.storage.get_summary()

        template = env.get_template("index.html")
        return template.render(runs=runs, failed_runs=failed_runs, summary=summary)

    def generate_comparison_report(
        self,
        tag: str,
        before_value: str,
        after_value: str,
        before_label: str | None = None,
        after_label: str | None = None,
    ) -> str:
        """Generate comparison HTML between two tag values.

        Args:
            tag: The tag key to compare on.
            before_value: The baseline tag value.
            after_value: The candidate tag value.
            before_label: Display label for baseline (defaults to before_value).
            after_label: Display label for candidate (defaults to after_value).

        Returns:
            HTML content as a string.
        """
        from .compare import compare_runs

        env = self._get_env()
        result = compare_runs(
            self.storage,
            tag=tag,
            before_value=before_value,
            after_value=after_value,
            before_label=before_label,
            after_label=after_label,
        )

        all_states = sorted(
            set(result.terminal_states_before.keys()) | set(result.terminal_states_after.keys())
        )
        all_tools = sorted(
            set(result.tool_usage_before.keys()) | set(result.tool_usage_after.keys())
        )

        template = env.get_template("comparison.html")
        return template.render(
            tag=result.tag,
            before_label=result.before_label,
            after_label=result.after_label,
            before_runs=result.before_runs,
            after_runs=result.after_runs,
            runs_delta=result.after_runs - result.before_runs,
            before_pass_rate=result.before_pass_rate,
            after_pass_rate=result.after_pass_rate,
            pass_rate_delta=result.pass_rate_delta,
            before_avg_turns=result.before_avg_turns,
            after_avg_turns=result.after_avg_turns,
            avg_turns_delta=result.avg_turns_delta,
            terminal_states_before=result.terminal_states_before,
            terminal_states_after=result.terminal_states_after,
            all_states=all_states,
            tool_usage_before=result.tool_usage_before,
            tool_usage_after=result.tool_usage_after,
            all_tools=all_tools,
            per_scene=result.per_scene,
        )

    def generate_static_report(self, output_path: Path | str) -> None:
        """Generate a complete static HTML report.

        Creates a directory structure:
            output_path/
                index.html
                runs/
                    {run_id}.html
                    ...

        Args:
            output_path: Directory to write the HTML files.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.static_mode = True
        self._env = None

        index_html = self.generate_index()
        (output_path / "index.html").write_text(index_html)

        runs_dir = output_path / "runs"
        runs_dir.mkdir(exist_ok=True)

        for run_id in self.storage.list_runs():
            run_html = self.generate_run_report(run_id)
            (runs_dir / f"{run_id}.html").write_text(run_html)

    def serve(self, port: int = 8080, host: str = "127.0.0.1") -> None:
        """Start a local HTTP server to browse reports.

        Args:
            port: Port to serve on.
            host: Host to bind to.
        """
        try:
            from http.server import BaseHTTPRequestHandler, HTTPServer
        except ImportError as e:
            raise ImportError("http.server module required") from e

        generator = self

        class ReportHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                from urllib.parse import parse_qs, urlparse

                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/" or path == "/index.html":
                    content = generator.generate_index()
                elif path == "/compare":
                    tag = query.get("tag", [None])[0]
                    before = query.get("before", [None])[0]
                    after = query.get("after", [None])[0]
                    if not all([tag, before, after]):
                        self.send_error(400, "Missing required params: tag, before, after")
                        return
                    try:
                        content = generator.generate_comparison_report(
                            tag=tag,
                            before_value=before,
                            after_value=after,
                            before_label=query.get("before_label", [None])[0],
                            after_label=query.get("after_label", [None])[0],
                        )
                    except ValueError as e:
                        self.send_error(400, str(e))
                        return
                elif path.startswith("/run/"):
                    run_id = path[5:]
                    try:
                        content = generator.generate_run_report(run_id)
                    except FileNotFoundError:
                        self.send_error(404, f"Run not found: {run_id}")
                        return
                else:
                    self.send_error(404, "Not found")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content.encode("utf-8"))

            def log_message(self, format: str, *args: Any) -> None:
                pass

        server = HTTPServer((host, port), ReportHandler)
        print(f"Serving reports at http://{host}:{port}")
        print("Press Ctrl+C to stop")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
