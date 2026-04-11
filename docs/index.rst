understudy
==========

**Simulated user testing for AI agents.**

Test your agents with synthetic conversations before they meet real users.

.. code-block:: bash

   pip install understudy[all]

.. toctree::
   :hidden:

   installation
   quickstart
   adk-integration
   langgraph-integration
   tutorial/index
   examples/index
   api/index

How It Works
------------

Testing with understudy is **4 steps**:

1. **Wrap your agent** — Adapt your agent (ADK, LangGraph, HTTP) to understudy's interface
2. **Mock your tools** — Register handlers that return test data instead of calling real services
3. **Write scenes** — YAML files defining what the simulated user wants and what you expect
4. **Run and assert** — Execute simulations, check traces, generate reports

The key insight: **assert against the trace, not the prose**. Don't check if the agent said "I've processed your return." Check if it called ``create_return()`` with the right arguments.

See It In Action
----------------

Browse real examples from the repo:

- `Example scene <https://github.com/gojiplus/understudy/blob/main/examples/scenes/return_eligible_backpack.yaml>`_ — YAML defining what the simulated user wants
- `ADK test file <https://github.com/gojiplus/understudy/blob/main/examples/adk/test_adk_returns.py>`_ — pytest assertions against traces
- `LangGraph test file <https://github.com/gojiplus/understudy/blob/main/examples/langgraph/test_langgraph_returns.py>`_ — same tests, different framework

**What a simulation run looks like:**

.. code-block:: text

   === return_eligible_backpack ===
   Turns: 6
   Tool calls: ['lookup_order', 'get_return_policy', 'create_return']

     ✓ required_tool: lookup_order called
     ✓ required_tool: create_return called

   PASSED

Get Started
-----------

1. :doc:`installation` — Install understudy
2. :doc:`quickstart` — Write your first scene and run a simulation
3. :doc:`adk-integration` — Full guide for Google ADK agents
4. :doc:`langgraph-integration` — Full guide for LangGraph agents
5. :doc:`examples/index` — Complete working example
