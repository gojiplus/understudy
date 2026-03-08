# Customer Service Agent Example

This example demonstrates understudy with a complete customer service agent
that handles order inquiries and return requests.

## Setup

1. Install dependencies:

```bash
pip install understudy[all]
# or
uv add understudy[all]
```

2. Set your API key:

```bash
# For agent (using Gemini via ADK)
export GOOGLE_API_KEY=your-key

# For judges (using Claude or other providers via litellm)
export ANTHROPIC_API_KEY=your-key
```

## Running the Demo

### Standalone Simulation

```bash
python run_simulation.py
```

This runs all three scenes and shows trace information.

### Pytest Tests

```bash
pytest test_returns.py -v
```

Runs deterministic assertions against traces.

### Full Judge Evaluation

```bash
pytest test_with_judges.py -v
```

Runs all 7 pre-built rubrics against each scene.

## Scenes

| Scene | Description |
|-------|-------------|
| `return_eligible_backpack.yaml` | Customer returns a backpack (eligible). Agent should process. |
| `return_nonreturnable_earbuds.yaml` | Customer returns earbuds (non-returnable). Agent should deny. |
| `adversarial_policy_bypass.yaml` | Customer tries social engineering. Agent should hold firm. |

## Agent

The agent (`customer_service_agent.py`) is a Google ADK agent with these tools:

- `lookup_order` - Get order details
- `lookup_customer_orders` - Get orders by email
- `get_return_policy` - Check category return policy
- `create_return` - Create a return request
- `escalate_to_human` - Hand off to human agent

## What Gets Tested

**Deterministic checks** (trace-based):
- Required tools were called
- Forbidden tools were NOT called
- Correct terminal state reached

**LLM judges** (sampling + majority vote):
- Tool usage correctness
- Policy compliance
- Tone and empathy
- Adversarial robustness
- Task completion
- Factual grounding
- Instruction following
