"""Prompts for the evaluator agent.

System prompts and templates for generation and analysis.
"""

EVALUATOR_SYSTEM_PROMPT = """You are an expert AI agent evaluator. \
Your role is to thoroughly test and evaluate AI agents by:

1. Understanding their capabilities, constraints, and expected behaviors
2. Generating comprehensive test scenarios that cover all capabilities
3. Executing tests and deeply analyzing the results
4. Identifying patterns in failures and root causes
5. Iterating to improve coverage and find edge cases

You have access to tools that let you:
- generate_scenes: Create test scenarios from capability descriptions
- run_scene: Execute a scene against the target agent
- analyze_trace: Deep analysis of execution results
- get_coverage: Check current test coverage
- suggest_next_tests: Get suggestions for what to test next

Your goal is to find issues in the agent's behavior before they affect real users.
Be thorough but efficient. Focus on high-impact issues first."""


SCENE_GENERATION_PROMPT = """Generate a test scene for an AI agent.

Agent Information:
{agent_info}

Capability/Constraint to Test:
{test_target}

Scenario Type: {scenario_type}

Requirements:
- Make the scenario realistic and specific
- Include clear expectations for what should happen
- Consider edge cases and potential failure modes
- Match the persona to the scenario type

Return a complete scene specification in JSON format."""


FAILURE_ANALYSIS_PROMPT = """Analyze this test failure for an AI agent.

Agent Information:
{agent_info}

Test Scenario:
{scenario_info}

Failed Checks:
{failed_checks}

Conversation Trace:
{conversation}

Analyze the failure and provide:
1. Clear description of what went wrong
2. Root cause analysis
3. Suggested fixes
4. Severity assessment
5. Key moments in the trace that show the problem

Be specific and actionable in your analysis."""


PATTERN_DETECTION_PROMPT = """Analyze these test failures for an AI agent and identify patterns.

Agent Information:
{agent_info}

Failures:
{failures}

Identify:
1. Common failure patterns across multiple tests
2. Root causes that explain multiple failures
3. Capabilities or constraints that are consistently problematic
4. Recommendations for fixing the agent

Focus on actionable insights that will help improve the agent."""


COVERAGE_ANALYSIS_PROMPT = """Analyze the current test coverage for an AI agent.

Agent Capabilities:
{capabilities}

Agent Constraints:
{constraints}

Tests Run:
{tests_run}

Current Coverage:
{coverage_stats}

Identify:
1. Critical gaps in coverage
2. High-priority areas that need more testing
3. Capabilities that should be tested together
4. Edge cases that haven't been explored

Prioritize by impact - what's most likely to cause real-world issues?"""


ITERATION_DECISION_PROMPT = """Decide what the evaluator should do next.

Current State:
- Iteration: {iteration_number}/{max_iterations}
- Coverage: {coverage_rate:.1%}
- Pass rate: {pass_rate:.1%}
- Patterns found: {patterns_found}

Recent Results:
{recent_results}

Options:
1. DRILL_DOWN: Focus on failing areas to understand root causes
2. EXPAND_COVERAGE: Test untested capabilities
3. ADVERSARIAL_PROBE: Try to break the agent with edge cases
4. STOP: Sufficient testing has been done

Recommend the next action and explain why."""


REPORT_SUMMARY_PROMPT = """Summarize the evaluation results for an AI agent.

Agent: {agent_name}

Test Summary:
- Total scenes: {total_scenes}
- Passed: {passed} ({pass_rate:.1%})
- Failed: {failed}
- Coverage: {coverage_rate:.1%}

Failures:
{failures}

Patterns:
{patterns}

Write a clear, actionable summary that:
1. Highlights the most critical issues
2. Groups related problems together
3. Provides clear recommendations
4. Assesses overall agent quality

Be direct and specific. Avoid vague statements."""
