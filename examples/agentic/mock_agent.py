"""Mock agentic app for testing and demonstration."""

from typing import Any

from understudy.agentic import StepResult, Task


class MockAgenticApp:
    """A mock agentic app that follows predefined behavior sequences.

    This allows testing the agentic evaluation framework without needing
    a real LLM-powered agent. Behaviors are defined as sequences of steps
    that the agent will execute for each task description.
    """

    def __init__(self, behaviors: dict[str, list[dict[str, Any]]]):
        """Initialize with predefined behaviors.

        Args:
            behaviors: Mapping from task descriptions to step sequences.
                       Each step is a dict with keys matching StepResult fields.
        """
        self.behaviors = behaviors
        self.task: Task | None = None
        self.environment: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.current_step = 0
        self._done = False
        self._outcome = "pending"
        self._state: dict[str, Any] = {}

    def start(self, task: Task, environment: dict[str, Any] | None = None) -> None:
        """Initialize the agent with a task."""
        self.task = task
        self.environment = environment or {}
        self.steps = self.behaviors.get(task.description, [])
        self.current_step = 0
        self._done = False
        self._outcome = "pending"
        self._state = dict(task.initial_state)

    def step(self) -> StepResult:
        """Execute one step from the predefined sequence."""
        if self.current_step >= len(self.steps):
            self._done = True
            self._outcome = "success"
            return StepResult(step_type="observe", observation="Task completed")

        step_data = self.steps[self.current_step]
        self.current_step += 1

        if "state_update" in step_data:
            self._state.update(step_data["state_update"])

        if step_data.get("marks_done"):
            self._done = True
            self._outcome = step_data.get("outcome", "success")

        return StepResult(
            step_type=step_data.get("step_type", "observe"),
            reasoning=step_data.get("reasoning"),
            action=step_data.get("action"),
            action_args=step_data.get("action_args"),
            observation=step_data.get("observation"),
            error=step_data.get("error"),
            tokens_used=step_data.get("tokens_used", 100),
        )

    def is_done(self) -> bool:
        """Check if the agent has completed."""
        return self._done

    def get_outcome(self) -> str:
        """Get the final outcome status."""
        return self._outcome

    def get_state(self) -> dict[str, Any]:
        """Get the current state."""
        return self._state

    def stop(self) -> None:
        """Clean up."""
        pass


def create_code_review_behavior() -> list[dict[str, Any]]:
    """Behavior sequence for code review task."""
    return [
        {
            "step_type": "think",
            "reasoning": "I need to read the target file first to review it",
            "tokens_used": 50,
        },
        {
            "step_type": "act",
            "action": "read_file",
            "action_args": {"path": "src/utils.py"},
            "tokens_used": 100,
        },
        {
            "step_type": "think",
            "reasoning": "I found several issues: using == None instead of 'is None', "
            "using range(len()) instead of enumerate, and import inside function",
            "tokens_used": 150,
        },
        {
            "step_type": "act",
            "action": "analyze_code",
            "action_args": {
                "content": "def process_data(x)...",
                "checks": ["style", "performance", "best_practices"],
            },
            "tokens_used": 200,
        },
        {
            "step_type": "think",
            "reasoning": "Analysis complete. I'll compile the findings into a report",
            "tokens_used": 80,
        },
        {
            "step_type": "act",
            "action": "write_report",
            "action_args": {
                "findings": [
                    {
                        "severity": "warning",
                        "line": 2,
                        "issue": "Use 'is None' instead of '== None'",
                    },
                    {
                        "severity": "info",
                        "line": 5,
                        "issue": "Use enumerate() instead of range(len())",
                    },
                    {
                        "severity": "warning",
                        "line": 10,
                        "issue": "Import statement inside function",
                    },
                ]
            },
            "state_update": {"report_generated": True},
            "marks_done": True,
            "outcome": "success",
            "tokens_used": 150,
        },
    ]


def create_file_search_behavior() -> list[dict[str, Any]]:
    """Behavior sequence for file search task."""
    return [
        {
            "step_type": "think",
            "reasoning": "I need to list Python files in src/ directory first",
            "tokens_used": 50,
        },
        {
            "step_type": "act",
            "action": "list_directory",
            "action_args": {"path": "src/", "pattern": "*.py"},
            "tokens_used": 80,
        },
        {
            "step_type": "think",
            "reasoning": "Found files: api.py, utils.py, client.py, models.py. "
            "Now I'll search each for 'import requests'",
            "tokens_used": 100,
        },
        {
            "step_type": "act",
            "action": "search_content",
            "action_args": {"pattern": "import requests", "files": ["src/*.py"]},
            "tokens_used": 150,
        },
        {
            "step_type": "think",
            "reasoning": "Found matches in src/api.py (line 1) and src/client.py (line 1)",
            "tokens_used": 80,
        },
        {
            "step_type": "observe",
            "observation": "Search complete",
            "state_update": {
                "results": [
                    {"file": "src/api.py", "line": 1},
                    {"file": "src/client.py", "line": 1},
                ],
                "found_count": 2,
            },
            "marks_done": True,
            "outcome": "success",
            "tokens_used": 50,
        },
    ]


def create_data_analysis_behavior() -> list[dict[str, Any]]:
    """Behavior sequence for data analysis task."""
    return [
        {
            "step_type": "think",
            "reasoning": "I need to read the CSV file to analyze the sales data",
            "tokens_used": 50,
        },
        {
            "step_type": "act",
            "action": "read_file",
            "action_args": {"path": "data/sales.csv"},
            "tokens_used": 100,
        },
        {
            "step_type": "think",
            "reasoning": "CSV has 5 rows of sales data. I'll compute total sales, "
            "average, and find the top product",
            "tokens_used": 120,
        },
        {
            "step_type": "act",
            "action": "compute_stats",
            "action_args": {
                "operations": ["sum", "mean", "group_by"],
                "columns": ["quantity", "price"],
            },
            "tokens_used": 200,
        },
        {
            "step_type": "think",
            "reasoning": "Calculated: Total=1425.00, Count=5, Top product=Widget A (25 units)",
            "tokens_used": 80,
        },
        {
            "step_type": "act",
            "action": "write_output",
            "action_args": {
                "path": "output/summary.json",
                "format": "json",
                "data": {
                    "total_sales": 1425.00,
                    "transaction_count": 5,
                    "average_per_transaction": 285.00,
                    "top_product": "Widget A",
                },
            },
            "state_update": {
                "total_sales": 1425.00,
                "transaction_count": 5,
                "top_product": "Widget A",
            },
            "marks_done": True,
            "outcome": "success",
            "tokens_used": 150,
        },
    ]


def create_mock_app() -> MockAgenticApp:
    """Create a mock app with all predefined behaviors."""
    behaviors = {
        "Review the Python file src/utils.py for code quality issues": (
            create_code_review_behavior()
        ),
        "Find all Python files that import 'requests' in the project": (
            create_file_search_behavior()
        ),
        "Analyze the sales data in data/sales.csv and compute summary statistics": (
            create_data_analysis_behavior()
        ),
    }
    return MockAgenticApp(behaviors=behaviors)
