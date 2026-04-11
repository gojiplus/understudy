# Understudy Examples

## Two Evaluation Paradigms

Understudy supports two distinct evaluation paradigms for AI agents:

### Conversational Agent Evaluation

Simulate multi-turn conversations with personas to test dialogue agents.

- **Use case**: Customer service bots, assistants, chatbots
- **Examples**: `adk/`, `langgraph/`
- **Scenes**: `scenes/*.yaml`

The simulated user follows a conversation plan, and you assert against the trace (tool calls, not prose).

### Agentic Flow Evaluation

Evaluate autonomous agents executing multi-step tasks.

- **Use case**: Code agents, research agents, task automation
- **Examples**: `agentic/`
- **Scenes**: `agentic_scenes/*.yaml`

The agent receives a task and goal, then autonomously executes actions. You assert against actions performed and outcomes.

## Directory Structure

```
examples/
├── scenes/              # Conversational scenes (shared by adk/langgraph)
├── agentic_scenes/      # Agentic task scenes
├── adk/                 # Google ADK conversational example
├── langgraph/           # LangGraph conversational example
├── agentic/             # Agentic flow example
└── README.md
```

## Running the Examples

### Conversational (ADK)

```bash
pip install understudy[adk]
export GOOGLE_API_KEY=your-key

cd examples/adk
python run_simulation.py
# or
pytest test_adk_returns.py -v
```

**Artifacts:**
- Traces: `.understudy/runs/`
- JUnit XML: `test-results/`

### Conversational (LangGraph)

```bash
pip install understudy[langgraph,reports]
export OPENAI_API_KEY=your-key

cd examples/langgraph
python run_simulation.py
# or
pytest test_langgraph_returns.py -v
```

**Artifacts:**
- Traces: `.understudy/langgraph_runs/`
- JUnit XML: `test-results/`
- HTML report: `report/index.html`

### Agentic

```bash
pip install understudy[all]

cd examples/agentic
python run_simulation.py
# or
pytest test_agentic.py -v
```

**Artifacts:**
- Traces: `.understudy/agentic_runs/{scene_id}/trace.json`

## Scene Examples

### Conversational Scene (`scenes/return_eligible_backpack.yaml`)

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
```

### Agentic Scene (`agentic_scenes/code_review_task.yaml`)

```yaml
id: code_review_task
description: Review Python code for issues

task:
  description: Review src/utils.py for code quality issues
  goal: Produce a report with style, performance, and best practice findings

environment:
  files:
    - src/utils.py

expectations:
  required_actions:
    - read_file
    - analyze_code
    - write_report
  forbidden_actions:
    - write_file
    - delete_file
  outcome: success
```

## Key Differences

| Aspect | Conversational | Agentic |
|--------|---------------|---------|
| **Interaction** | Multi-turn dialogue | Single task execution |
| **Simulated party** | User with persona | None (agent acts alone) |
| **Scene defines** | Conversation plan | Task and goal |
| **Assert on** | `trace.called("tool")` | `trace.performed("action")` |
| **Termination** | Max turns or agent ends | Task complete or max steps |
