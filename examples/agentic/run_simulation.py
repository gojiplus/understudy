#!/usr/bin/env python
"""Standalone demo script for agentic evaluation.

Run with:
    cd examples/agentic
    python run_simulation.py
"""

import sys
from pathlib import Path

from mock_agent import create_mock_app

from understudy.agentic import AgenticScene, check_agentic, run_agentic
from understudy.mocks import MockToolkit, ToolError


def create_mocks() -> MockToolkit:
    """Create mock handlers for the agentic tools."""
    toolkit = MockToolkit()

    files = {
        "src/utils.py": (
            "def process_data(x):\n"
            "    if x == None:\n"
            "        return []\n"
            "    result = []\n"
            "    for i in range(len(x)):\n"
            "        result.append(x[i] * 2)\n"
            "    return result\n"
            "\n"
            "def fetch_user(id):\n"
            "    import requests\n"
            "    r = requests.get(f'http://api.example.com/users/{id}')\n"
            "    return r.json()\n"
        ),
        "src/api.py": (
            "import requests\n"
            "import json\n"
            "\n"
            "def call_api(url):\n"
            "    return requests.get(url).json()\n"
        ),
        "src/client.py": (
            "import requests\n"
            "from typing import Optional\n"
            "\n"
            "class Client:\n"
            "    def fetch(self, url: str) -> Optional[dict]:\n"
            "        return requests.get(url).json()\n"
        ),
        "src/models.py": (
            "from dataclasses import dataclass\n"
            "\n"
            "@dataclass\n"
            "class User:\n"
            "    name: str\n"
        ),
        "data/sales.csv": (
            "date,product,quantity,price\n"
            "2025-01-01,Widget A,10,25.00\n"
            "2025-01-02,Widget B,5,50.00\n"
            "2025-01-03,Widget A,15,25.00\n"
            "2025-01-04,Widget C,3,100.00\n"
            "2025-01-05,Widget B,8,50.00\n"
        ),
    }

    directories = {
        "src/": ["api.py", "utils.py", "client.py", "models.py"],
        "data/": ["sales.csv"],
    }

    @toolkit.handle("read_file")
    def read_file(path: str) -> str:
        if path in files:
            return files[path]
        raise ToolError(f"File not found: {path}")

    @toolkit.handle("list_directory")
    def list_directory(path: str, pattern: str | None = None) -> list[str]:
        if path in directories:
            result = directories[path]
            if pattern and pattern.endswith(".py"):
                result = [f for f in result if f.endswith(".py")]
            return result
        raise ToolError(f"Directory not found: {path}")

    @toolkit.handle("search_content")
    def search_content(pattern: str, file_paths: list[str] | None = None) -> list[dict]:
        _ = file_paths
        results = []
        for path, content in files.items():
            if pattern in content:
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern in line:
                        results.append({"file": path, "line": i, "content": line.strip()})
        return results

    @toolkit.handle("analyze_code")
    def analyze_code(content: str, checks: list[str] | None = None) -> dict:
        _ = content, checks
        return {
            "issues": [
                {"type": "style", "message": "Use 'is None' instead of '== None'"},
                {"type": "performance", "message": "Use enumerate() instead of range(len())"},
                {"type": "best_practice", "message": "Avoid imports inside functions"},
            ],
            "score": 6.5,
        }

    @toolkit.handle("write_report")
    def write_report(findings: list[dict]) -> dict:
        return {"status": "written", "finding_count": len(findings)}

    @toolkit.handle("compute_stats")
    def compute_stats(operations: list[str], columns: list[str]) -> dict:
        _ = operations, columns
        return {
            "total": 1425.00,
            "mean": 285.00,
            "count": 5,
            "by_product": {"Widget A": 25, "Widget B": 13, "Widget C": 3},
        }

    @toolkit.handle("write_output")
    def write_output(path: str, format: str, data: dict) -> dict:
        _ = data
        return {"status": "written", "path": path, "format": format}

    return toolkit


def main():
    app = create_mock_app()
    mocks = create_mocks()

    scenes_dir = Path(__file__).parent.parent / "agentic_scenes"
    scene_files = list(scenes_dir.glob("*.yaml"))

    if not scene_files:
        print("No scene files found in agentic_scenes/")
        sys.exit(1)

    print(f"Running {len(scene_files)} agentic scenes...\n")

    output_dir = Path(".understudy/agentic_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True
    for scene_file in sorted(scene_files):
        scene = AgenticScene.from_file(scene_file)

        print(f"{'=' * 60}")
        print(f"Scene: {scene.id}")
        print(f"Task: {scene.task.description[:50]}...")
        print(f"Goal: {scene.task.goal[:50]}...")

        trace = run_agentic(app, scene, mocks=mocks)

        print(f"\nOutcome: {trace.outcome}")
        print(f"Steps: {trace.total_steps}")
        print(f"Tokens: {trace.total_tokens}")
        print(f"Actions: {trace.action_sequence()}")

        result = check_agentic(trace, scene.expectations)
        print(f"\n{result.summary()}")

        scene_output_dir = output_dir / scene.id
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = scene_output_dir / "trace.json"
        trace_file.write_text(trace.model_dump_json(indent=2))
        print(f"\nTrace saved to: {trace_file}")

        if result.passed:
            print("\nRESULT: PASSED")
        else:
            print("\nRESULT: FAILED")
            all_passed = False
        print()

    print("=" * 60)
    if all_passed:
        print("All scenes passed!")
        sys.exit(0)
    else:
        print("Some scenes failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
