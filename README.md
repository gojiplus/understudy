## understudy: Scenario Testing for AI Agents

[![PyPI version](https://badge.fury.io/py/understudy.svg)](https://badge.fury.io/py/understudy)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/understudy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/project/understudy)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://github.com/gojiplus/understudy/actions/workflows/docs.yml/badge.svg)](https://gojiplus.github.io/understudy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Understudy is a scenario-driven testing framework for AI agents that simulates realistic multi-turn users, runs those scenes against an agent through a simple app adapter, records a structured execution trace of messages, tool calls, and handoffs, and then evaluates behavior with deterministic checks, optional LLM judges, and run reports.

## How It Works

Testing with understudy is **4 steps**:

1. **Wrap your agent** — Adapt your agent (ADK, LangGraph, HTTP) to understudy's interface
2. **Mock your tools** — Register handlers that return test data instead of calling real services
3. **Write scenes** — YAML files defining what the simulated user wants and what you expect
4. **Run and assert** — Execute simulations, check traces, generate reports

The key insight: **assert against the trace, not the prose**. Don't check what the agent said—check what it did (tool calls).

**See real examples:**
- [Example scene](https://github.com/gojiplus/understudy/blob/main/example/scenes/return_eligible_backpack.yaml) — YAML defining a test scenario
- [ADK test file](https://github.com/gojiplus/understudy/blob/main/example/adk/test_returns.py) — pytest assertions against traces
- [LangGraph test file](https://github.com/gojiplus/understudy/blob/main/example/langgraph/test_returns.py) — same tests, different framework

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
  forbidden_tools:
    - issue_refund
```

### 4. Run simulation

```python
from understudy import Scene, run

scene = Scene.from_file("scenes/return_backpack.yaml")
trace = run(app, scene, mocks=mocks)

assert trace.called("lookup_order")
assert trace.called("create_return")
assert not trace.called("issue_refund")
```

Or with pytest (define `app` and `mocks` fixtures in conftest.py):

```bash
pytest test_returns.py -v
```

## Suites and Batch Runs

Run multiple scenes with multiple simulations per scene:

```python
from understudy import Suite, RunStorage

suite = Suite.from_directory("scenes/")
storage = RunStorage()

# Run each scene 3 times and tag for comparison
results = suite.run(
    app,
    mocks=mocks,
    storage=storage,
    n_sims=3,
    tags={"version": "v1"},
)
print(f"{results.pass_count}/{len(results.results)} passed")
```

## Simulation and Evaluation

Understudy separates simulation (generating traces) from evaluation (checking traces). Use together or separately:

### Combined (most common)

```bash
understudy run \
  --app mymodule:agent_app \
  --scene ./scenes/ \
  --n-sims 3 \
  --junit results.xml
```

### Separate workflows

Generate traces only:

```bash
understudy simulate \
  --app mymodule:agent_app \
  --scenes ./scenes/ \
  --output ./traces/ \
  --n-sims 3
```

Evaluate existing traces:

```bash
understudy evaluate \
  --traces ./traces/ \
  --output ./results/ \
  --junit results.xml
```

Python API:

```python
from understudy import simulate_batch, evaluate_batch

# Generate traces
traces = simulate_batch(
    app=agent_app,
    scenes="./scenes/",
    n_sims=3,
    output="./traces/",
)

# Evaluate later
results = evaluate_batch(
    traces="./traces/",
    output="./results/",
)
```

## CLI Commands

```bash
# Run simulations
understudy run --app mymodule:app --scene ./scenes/
understudy simulate --app mymodule:app --scenes ./scenes/
understudy evaluate --traces ./traces/

# View results
understudy list
understudy show <run_id>
understudy summary

# Compare runs by tag
understudy compare --tag version --before v1 --after v2

# Generate reports
understudy report -o report.html
understudy compare --tag version --before v1 --after v2 --html comparison.html

# Interactive browser
understudy serve --port 8080

# HTTP simulator server (for browser/UI testing)
understudy serve-api --port 8000

# Cleanup
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

**[View example report](https://htmlpreview.github.io/?https://github.com/gojiplus/understudy/blob/main/example/langgraph/report/index.html)**

The `understudy summary` command shows:
- **Pass rate** — percentage of scenes that passed all expectations
- **Avg turns** — average conversation length
- **Tool usage** — distribution of tool calls across runs
- **Agents** — which agents were invoked

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
- [LangGraph integration](https://gojiplus.github.io/understudy/langgraph-integration.html)
- [HTTP client for deployed agents](https://gojiplus.github.io/understudy/tutorial/http.html)
- [API reference](https://gojiplus.github.io/understudy/api/index.html)

## License

MIT
