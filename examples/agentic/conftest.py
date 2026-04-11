"""pytest fixtures for agentic example tests."""

from pathlib import Path

import pytest
from mock_agent import create_mock_app

from understudy.mocks import MockToolkit, ToolError


@pytest.fixture
def app():
    """Create a mock agentic app with predefined behaviors."""
    return create_mock_app()


@pytest.fixture
def scenes_dir():
    """Path to the agentic scenes directory."""
    return Path(__file__).parent.parent / "agentic_scenes"


@pytest.fixture
def mocks():
    """Create mock handlers for the agentic tools.

    These mocks simulate the tools that the agent would normally use.
    """
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
