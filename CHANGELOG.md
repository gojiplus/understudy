# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Isolated next-turn tests with synthetic state/history, native completed-turn
  checkpoints, state patches, prefix replay, and fresh app/mock factories.
- ADK app-wide tool interception and native nested invocation paths; LangGraph
  checkpoint forks and nested graph tracing; a LangChain create_agent adapter.
- `turn_runner` pytest fixture, `run-turn` and `replay-until` commands, and report
  details for next-turn boundaries, state changes, and agent invocations.

### Changed
- LangGraph mock context is isolated per execution. Parallel suites and batches
  require app factories. Graph completion no longer ends a conversation.
- Custom agentic adapters must intercept mocks before action execution; the runner
  no longer changes observations after a real action has run.
- Unsupported state, interception, or observation requests fail explicitly.


### Fixed

- HTTP adapters create remote sessions and use the documented ADK and Agent
  Engine request formats. Streamed tool results, transfers, termination, and
  token usage are captured from actual ADK event shapes.
- Supplied Agent Engine credentials are refreshed when invalid.
- ADK mock callbacks are restored after a run, and transfers are reported once
  per turn. LangGraph mocks are scoped to graph execution without modifying
  caller configuration, and standardized token usage is recorded.

### Changed

- Measure package-only line and branch coverage before pytest plugin imports,
  with an 80% floor and deterministic adapter and plugin regression tests.

- Adopted the py-canon fleet standard: shared reusable CI/docs/release
  workflows, canon ruff/pyright/pydoclint configuration, and a
  py-canon-managed Sphinx config.

## [0.5.0] - 2025-04-10

### Added

- **Agentic flow evaluation**: New paradigm for evaluating autonomous task-executing agents
  - `AgenticScene` for defining task-based evaluation scenarios
  - `AgenticTrace` for capturing action sequences and outcomes
  - `run_agentic()` and `check_agentic()` for running and validating agentic flows
  - New expectation types: `required_actions`, `forbidden_actions`, `outcome`
- **Agentic examples**: `examples/agentic/` with code review, file search, and data analysis tasks
- **Agentic scenes**: `examples/agentic_scenes/` with YAML task definitions

### Changed

- Renamed `example/` directory to `examples/` for consistency
- Renamed test files to unique names (`test_adk_returns.py`, `test_langgraph_returns.py`)

## [0.4.0] - 2025-03-15

### Added

- **Separate simulation and evaluation**: Decouple trace generation from evaluation
  - `understudy simulate` — Generate traces without evaluation
  - `understudy evaluate` — Evaluate existing traces against expectations
  - `understudy run` — Combined simulate + evaluate (existing behavior)
- **Multiple simulations per scene**: Run each scene multiple times with `--n-sims`
  - `Suite.run(n_sims=3)` for Python API
  - `--n-sims` flag for CLI commands
- **TraceStorage**: Dedicated storage for simulation-only traces
- **EvaluationStorage**: Dedicated storage for evaluation results
- **Python API functions**:
  - `simulate()` and `simulate_batch()` for simulation-only workflows
  - `evaluate()` and `evaluate_batch()` for trace evaluation

## [0.3.0] - 2025-03-14

### Added

- **LangGraph adapter**: Test LangGraph agents with `LangGraphApp`
- **HTTP simulator server**: Test browser-based and UI agents via HTTP
  - `understudy serve-api` command for HTTP simulator endpoints
- **Metrics system**: Compute and track metrics on traces
  - `efficiency`, `resolution_match`, `tool_trajectory` built-in metrics
  - Custom metrics via `MetricRegistry`
- **State snapshots**: Capture agent state at each turn for debugging

## [0.2.0] - 2025-03-13

### Added

- **Comparison reports**: Compare runs across different configurations using tags
  - `compare_runs()` function for programmatic comparison
  - `understudy compare` CLI command with `--html` option for HTML reports
- **Tag-based filtering**: Tag runs with metadata for grouping and comparison
  - `Suite.run(tags={"version": "v1"})` to tag all runs in a suite
  - `RunStorage.save(tags={"model": "gpt-4o"})` for individual run tagging
- **LiteLLMBackend**: Unified LLM backend using litellm
  - Supports OpenAI, Anthropic, Google, and 100+ other providers
  - Model strings: `"gpt-4o"`, `"claude-sonnet-4-20250514"`, `"gemini/gemini-1.5-flash"`, etc.
- **HTTP adapter**: Test deployed agents via HTTP endpoints
  - `HTTPAgentApp` for connecting to REST APIs

### Changed

- `litellm` is now a core dependency (previously optional under `[judges]`)
- Removed `[judges]` optional dependency group

## [0.1.0] - 2025-02-01

### Added

- Initial release
- Scene-based simulation framework
- Persona presets (cooperative, adversarial, frustrated)
- Trace recording and analysis
- Expectation-based assertions
- MockToolkit for tool mocking
- Judge system with LLM-based evaluation
- Suite runner for batch execution
- RunStorage for persisting results
- HTML report generation
- CLI for report viewing and run management
