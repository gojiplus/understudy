ADK Integration Guide
=====================

This guide explains how to test Google ADK agents with understudy.

Prerequisites
-------------

Install understudy with ADK support:

.. code-block:: bash

   pip install understudy[adk]

You'll need a Google API key:

.. code-block:: bash

   export GOOGLE_API_KEY=your-key-here

Wrapping Your Agent
-------------------

understudy wraps your ADK agent in an ``ADKApp`` adapter:

.. code-block:: python

   from google.adk import Agent
   from google.adk.tools import FunctionTool
   from understudy.adk import ADKApp

   # Define your agent
   agent = Agent(
       model="gemini-2.5-flash",
       name="customer_service",
       instruction="You are a customer service agent...",
       tools=[FunctionTool(lookup_order), FunctionTool(create_return)],
   )

   # Wrap it for understudy
   app = ADKApp(agent=agent)

Mocking Tool Responses
----------------------

Your agent's tools call external services. For testing, mock them with ``MockToolkit``:

.. code-block:: python

   from understudy.mocks import MockToolkit

   mocks = MockToolkit()

   @mocks.handle("lookup_order")
   def lookup_order(order_id: str) -> dict:
       orders = {
           "ORD-10031": {
               "order_id": "ORD-10031",
               "items": [{"name": "Hiking Backpack", "sku": "HB-220"}],
               "status": "delivered",
           }
       }
       if order_id not in orders:
           raise ToolError(f"Order {order_id} not found")
       return orders[order_id]

   @mocks.handle("create_return")
   def create_return(order_id: str, item_sku: str, reason: str) -> dict:
       return {"return_id": "RET-001", "status": "created"}

Pass mocks to ``run()``:

.. code-block:: python

   trace = run(app, scene, mocks=mocks)

Running Simulations
-------------------

With your agent wrapped and mocks defined:

.. code-block:: python

   from understudy import Scene, run

   scene = Scene.from_file("scenes/test_scenario.yaml")
   trace = run(app, scene, mocks=mocks)

   print(f"Tool calls: {trace.call_sequence()}")
   print(f"Terminal state: {trace.terminal_state}")

Subagent Tracing
----------------

understudy tracks which agent handled each turn in multi-agent ADK applications. This is useful for testing agent routing and delegation.

Inspecting Agent Attribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # See which agents were invoked
   print(trace.agents_invoked())  # ["customer_service", "billing_agent"]

   # Check if a specific agent called a tool
   assert trace.agent_called("billing_agent", "process_refund")

   # Get all calls made by a specific agent
   billing_calls = trace.calls_by_agent("billing_agent")

Agent Transfers
~~~~~~~~~~~~~~~

When agents hand off to other agents, understudy records the transfer:

.. code-block:: python

   for transfer in trace.agent_transfers:
       print(f"{transfer.from_agent} -> {transfer.to_agent}")

Setting Expectations
~~~~~~~~~~~~~~~~~~~~

Validate agent behavior with expectations:

.. code-block:: yaml

   # scene.yaml
   expectations:
     required_agents:
       - customer_service
       - billing_agent
     forbidden_agents:
       - admin_agent
     required_agent_tools:
       billing_agent:
         - process_refund

pytest Fixtures
---------------

Set up reusable fixtures in ``conftest.py``:

.. code-block:: python

   # conftest.py
   import pytest
   from understudy.adk import ADKApp
   from understudy.mocks import MockToolkit
   from my_agent import customer_service_agent

   @pytest.fixture
   def app():
       return ADKApp(agent=customer_service_agent)

   @pytest.fixture
   def mocks():
       toolkit = MockToolkit()

       @toolkit.handle("lookup_order")
       def lookup_order(order_id: str) -> dict:
           return {"order_id": order_id, "items": [...], "status": "delivered"}

       @toolkit.handle("create_return")
       def create_return(order_id: str, item_sku: str, reason: str) -> dict:
           return {"return_id": "RET-001", "status": "created"}

       return toolkit

Then use in tests:

.. code-block:: python

   # test_agent.py
   from understudy import Scene, run, check

   def test_return_flow(app, mocks):
       scene = Scene.from_file("scenes/return_backpack.yaml")
       trace = run(app, scene, mocks=mocks)
       results = check(trace, scene.expectations)
       assert results.passed, results.summary()

Full Example
------------

Here's a complete test file:

.. code-block:: python

   import pytest
   from understudy import Scene, Suite, run, check, Judge
   from understudy.adk import ADKApp
   from understudy.mocks import MockToolkit
   from my_agent import customer_service_agent

   @pytest.fixture
   def app():
       return ADKApp(agent=customer_service_agent)

   @pytest.fixture
   def mocks():
       toolkit = MockToolkit()

       @toolkit.handle("lookup_order")
       def lookup_order(order_id: str) -> dict:
           return {"order_id": order_id, "status": "delivered", "items": [...]}

       @toolkit.handle("create_return")
       def create_return(order_id: str, item_sku: str, reason: str) -> dict:
           return {"return_id": "RET-001", "status": "created"}

       return toolkit

   def test_basic_return(app, mocks):
       """Test that returnable items can be returned."""
       scene = Scene.from_file("scenes/return_eligible_backpack.yaml")
       trace = run(app, scene, mocks=mocks)

       assert trace.called("lookup_order")
       assert trace.called("create_return")

   def test_policy_enforcement(app, mocks):
       """Test that non-returnable items are denied."""
       scene = Scene.from_file("scenes/return_nonreturnable_earbuds.yaml")
       trace = run(app, scene, mocks=mocks)

       assert not trace.called("create_return")

   def test_with_judge(app, mocks):
       """Use LLM judge for soft checks."""
       scene = Scene.from_file("scenes/adversarial_bypass.yaml")
       trace = run(app, scene, mocks=mocks)

       judge = Judge(rubric="Agent remained firm on policy despite pressure.")
       result = judge.evaluate(trace)
       assert result.score == 1

   def test_full_suite(app, mocks):
       """Run all scenes."""
       suite = Suite.from_directory("scenes/")
       results = suite.run(app, mocks=mocks)
       assert results.all_passed, results.summary()

Troubleshooting
---------------

**ImportError: google-adk package required**

Install the ADK extra:

.. code-block:: bash

   pip install understudy[adk]

**Tools returning None**

Make sure you've registered mock handlers for all tools your agent uses. Check which tools are being called:

.. code-block:: python

   print(trace.call_sequence())

Then ensure each tool has a corresponding ``@mocks.handle()`` decorator.
