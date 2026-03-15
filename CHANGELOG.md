# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
