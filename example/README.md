# Examples

## Structure

```
example/
├── scenes/          # YAML scenarios (shared by all frameworks)
├── adk/             # Google ADK example
└── langgraph/       # LangGraph example
```

## Scenes

The `scenes/` directory contains test scenarios:

| Scene | Description |
|-------|-------------|
| `return_eligible_backpack.yaml` | Customer returns a backpack (allowed) |
| `return_nonreturnable_earbuds.yaml` | Customer returns earbuds (denied) |
| `adversarial_policy_bypass.yaml` | Social engineering attempt |

## Running Examples

### ADK

```bash
pip install understudy[adk]
export GOOGLE_API_KEY=your-key
cd example/adk
pytest test_returns.py -v
```

### LangGraph

```bash
pip install understudy[langgraph]
export OPENAI_API_KEY=your-key
cd example/langgraph
pytest test_returns.py -v
```

## Key Point

Both examples use the **same scenes** and **same assertion patterns**:

```python
assert trace.called("lookup_order")
assert trace.called("get_return_policy")
assert not trace.called("create_return")  # Should be denied
```

The framework adapter (`ADKApp` vs `LangGraphApp`) is the only difference.
