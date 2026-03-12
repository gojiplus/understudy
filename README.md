## understudy: Scenario Testing for AI Agents

[![PyPI version](https://badge.fury.io/py/understudy.svg)](https://badge.fury.io/py/understudy)
[![Downloads](https://pepy.tech/badge/understudy)](https://pepy.tech/project/understudy)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://github.com/gojiplus/understudy/actions/workflows/docs.yml/badge.svg)](https://gojiplus.github.io/understudy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Understudy is a scenario-driven testing framework for AI agents that simulates realistic multi-turn users, runs those scenes against an agent through a simple app adapter, records a structured execution trace of messages, tool calls, handoffs, and terminal states, and then evaluates behavior with deterministic checks, optional LLM judges, and run reports.

## How It Works

Testing with understudy is **4 steps**:

1. **Wrap your agent** — Adapt your agent (ADK, LangGraph, HTTP) to understudy's interface
2. **Mock your tools** — Register handlers that return test data instead of calling real services
3. **Write scenes** — YAML files defining what the simulated user wants and what you expect
4. **Run and assert** — Execute simulations, check traces, generate reports

The key insight: **assert against the trace, not the prose**. Don't check what the agent said—check what it did (tool calls, terminal state).

**See real examples:**
- [Example scene](https://github.com/gojiplus/understudy/blob/main/example/scenes/return_eligible_backpack.yaml) — YAML defining a test scenario
- [Test file](https://github.com/gojiplus/understudy/blob/main/example/test_returns.py) — pytest assertions against traces
- [Sample report](https://htmlpreview.github.io/?https://github.com/gojiplus/understudy/blob/main/example/sample_report.html) — HTML output from `understudy report`

## Installation

```bash
pip install understudy[all]
```

## Quick Start

### 1. Wrap your agent

```python
from understudy.adk import ADKApp
from my_agent import agent

app = ADKApp(agent=agent)
```

### 2. Mock your tools

Your agent has tools that call external services. Mock them for testing:

```python
from understudy.mocks import MockToolkit

mocks = MockToolkit()

@mocks.handle("lookup_order")
def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "items": [...], "status": "delivered"}

@mocks.handle("create_return")
def create_return(order_id: str, item_sku: str, reason: str) -> dict:
    return {"return_id": "RET-001", "status": "created"}
```

### 3. Write a scene

Create `scenes/return_backpack.yaml`:

```yaml
id: return_eligible_backpack
description: Customer wants to return a backpack

starting_prompt: "I'd like to return an item please."
conversation_plan: |
  Goal: Return the hiking backpack from order ORD-10031.
  - Provide order ID when asked
  - Return reason: too small

persona: cooperative
max_turns: 15

expectations:
  required_tools:
    - lookup_order
    - create_return
  allowed_terminal_states:
    - return_created
```

### 4. Run simulation

```python
from understudy import Scene, run, check

scene = Scene.from_file("scenes/return_backpack.yaml")
trace = run(app, scene, mocks=mocks)

assert trace.called("lookup_order")
assert trace.called("create_return")
assert trace.terminal_state == "return_created"
```

Or with pytest (define `app` and `mocks` fixtures in conftest.py):

```bash
pytest test_returns.py -v
```

## CLI Commands

After running simulations, use the CLI to inspect results:

```bash
# List all saved runs
understudy list

# Show aggregate metrics (pass rate, avg turns, tool usage, terminal states)
understudy summary

# Show details for a specific run
understudy show <run_id>

# Generate static HTML report
understudy report --output report.html

# Start interactive report browser
understudy serve --port 8080

# Delete runs
understudy delete <run_id>
understudy clear
```

## LLM Judges

For qualities that can't be checked deterministically:

```python
from understudy.judges import Judge

empathy_judge = Judge(
    rubric="The agent acknowledged frustration and was empathetic while enforcing policy.",
    samples=5,
)

result = empathy_judge.evaluate(trace)
assert result.score == 1
```

Built-in rubrics:

```python
from understudy.judges import (
    TOOL_USAGE_CORRECTNESS,
    POLICY_COMPLIANCE,
    TONE_EMPATHY,
    ADVERSARIAL_ROBUSTNESS,
    TASK_COMPLETION,
)
```

## Report Contents

The `understudy summary` command shows:
- **Pass rate** - percentage of scenes that passed all expectations
- **Avg turns** - average conversation length
- **Tool usage** - distribution of tool calls across runs
- **Terminal states** - breakdown of how conversations ended
- **Agents** - which agents were invoked

The HTML report (`understudy report`) includes:
- All metrics above
- Full conversation transcripts
- Tool call details with arguments
- Expectation check results
- Judge evaluation results (when used)

## Documentation

See the [full documentation](https://gojiplus.github.io/understudy) for:
- [Installation guide](https://gojiplus.github.io/understudy/installation.html)
- [Writing scenes](https://gojiplus.github.io/understudy/tutorial/scenes.html)
- [ADK integration](https://gojiplus.github.io/understudy/adk-integration.html)
- [HTTP client for deployed agents](https://gojiplus.github.io/understudy/tutorial/http.html)
- [API reference](https://gojiplus.github.io/understudy/api/index.html)

## License

MIT
