"""Reports: generate HTML reports from saved runs."""

from pathlib import Path
from typing import Any

from .storage import RunStorage


class ReportGenerator:
    """Generate HTML reports from saved simulation runs."""

    def __init__(self, storage: RunStorage):
        """
        Args:
            storage: RunStorage instance containing saved runs.
        """
        self.storage = storage
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
        )

    def generate_index(self) -> str:
        """Generate summary HTML listing all runs.

        Returns:
            HTML content as a string.
        """
        env = self._get_env()
        runs = []
        for run_id in self.storage.list_runs():
            data = self.storage.load(run_id)
            meta = data.get("metadata", {})
            runs.append(
                {
                    "run_id": run_id,
                    "scene_id": meta.get("scene_id", run_id),
                    "passed": meta.get("passed"),
                    "terminal_state": meta.get("terminal_state"),
                    "turn_count": meta.get("turn_count", 0),
                    "tools_called": meta.get("tools_called", []),
                    "timestamp": meta.get("timestamp", ""),
                }
            )

        summary = self.storage.get_summary()

        template = env.get_template("index.html")
        return template.render(runs=runs, summary=summary)

    def generate_metrics_report(self) -> str:
        """Generate aggregate metrics HTML.

        Returns:
            HTML content as a string.
        """
        env = self._get_env()
        summary = self.storage.get_summary()

        runs = self.storage.load_all()
        judge_agreement = self._compute_judge_agreement(runs)

        template = env.get_template("metrics.html")
        return template.render(summary=summary, judge_agreement=judge_agreement)

    def _compute_judge_agreement(self, runs: list[dict]) -> dict[str, dict[str, float]]:
        """Compute average agreement rates and pass rates for each judge rubric."""
        rubric_stats: dict[str, dict[str, list]] = {}

        for run in runs:
            judges = run.get("judges", {})
            for name, result in judges.items():
                if name not in rubric_stats:
                    rubric_stats[name] = {"agreements": [], "scores": []}

                agreement = result.get("agreement_rate")
                if agreement is not None:
                    rubric_stats[name]["agreements"].append(agreement)

                score = result.get("score")
                if score is not None:
                    rubric_stats[name]["scores"].append(score)

        result = {}
        for name, stats in rubric_stats.items():
            agreements = stats["agreements"]
            scores = stats["scores"]
            result[name] = {
                "agreement": sum(agreements) / len(agreements) if agreements else 0.0,
                "pass_rate": sum(scores) / len(scores) if scores else 0.0,
            }

        return result

    def generate_static_report(self, output_path: Path | str) -> None:
        """Generate a complete static HTML report.

        Args:
            output_path: Path to write the HTML file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self.generate_index()
        output_path.write_text(html)

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
                path = self.path

                if path == "/" or path == "/index.html":
                    content = generator.generate_index()
                elif path == "/metrics":
                    content = generator.generate_metrics_report()
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
