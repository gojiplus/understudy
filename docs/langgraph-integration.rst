LangGraph Integration Guide
===========================

This guide explains how to test LangGraph agents with understudy.

Prerequisites
-------------

Install understudy with LangGraph support:

.. code-block:: bash

   pip install understudy[langgraph]

You'll need an API key for your LLM provider:

.. code-block:: bash

   export OPENAI_API_KEY=your-key-here

Wrapping Your Agent
-------------------

understudy wraps your LangGraph agent in a ``LangGraphApp`` adapter:

.. code-block:: python

   from langchain_openai import ChatOpenAI
   from langgraph.graph import StateGraph, END
   from understudy.langgraph import LangGraphApp

   # Define your graph
   model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
   graph = create_my_agent(model)

   # Wrap it for understudy
   app = LangGraphApp(graph=graph)

Mockable Tools
--------------

LangGraph tools need the ``@mockable_tool`` decorator to work with understudy's mock system:

.. code-block:: python

   from langchain_core.tools import tool
   from understudy.langgraph.tools import mockable_tool

   @tool
   @mockable_tool
   def lookup_order(order_id: str) -> dict:
       """Look up an order by ID."""
       raise NotImplementedError("Use mocks in tests")

   @tool
   @mockable_tool
   def create_return(order_id: str, item_sku: str, reason: str) -> dict:
       """Create a return request."""
       raise NotImplementedError("Use mocks in tests")

The decorator enables understudy to intercept tool calls during simulation.

Mocking Tool Responses
----------------------

Register mock handlers with ``MockToolkit``:

.. code-block:: python

   from understudy.mocks import MockToolkit, ToolError

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

Running Simulations
-------------------

LangGraph requires ``MockableToolContext`` to route tool calls through mocks:

.. code-block:: python

   from understudy import Scene, run
   from understudy.langgraph.tools import MockableToolContext

   scene = Scene.from_file("scenes/test_scenario.yaml")

   with MockableToolContext(mocks):
       trace = run(app, scene, mocks=mocks)

   print(f"Tool calls: {trace.call_sequence()}")
   print(f"Terminal state: {trace.terminal_state}")

The context manager activates mock routing for the duration of the simulation.

State Snapshots
---------------

LangGraph maintains state between turns. understudy captures state snapshots in the trace:

.. code-block:: python

   # Access the final state
   print(trace.final_state)

   # Inspect state at each turn
   for turn in trace.turns:
       if hasattr(turn, 'state'):
           print(turn.state)

pytest Fixtures
---------------

Set up reusable fixtures in ``conftest.py``:

.. code-block:: python

   # conftest.py
   import pytest
   from langchain_openai import ChatOpenAI
   from understudy.langgraph import LangGraphApp
   from understudy.mocks import MockToolkit, ToolError
   from my_agent import create_customer_service_agent, tools

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

   @pytest.fixture
   def app(mocks):
       model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
       graph = create_customer_service_agent(model)
       return LangGraphApp(graph=graph)

Then use in tests:

.. code-block:: python

   # test_agent.py
   from understudy import Scene, run, check
   from understudy.langgraph.tools import MockableToolContext

   def test_return_flow(app, mocks):
       scene = Scene.from_file("scenes/return_backpack.yaml")

       with MockableToolContext(mocks):
           trace = run(app, scene, mocks=mocks)

       results = check(trace, scene.expectations)
       assert results.passed, results.summary()

Full Example
------------

Here's a complete test file:

.. code-block:: python

   import pytest
   from understudy import Scene, Suite, run, check, Judge
   from understudy.langgraph import LangGraphApp
   from understudy.langgraph.tools import MockableToolContext
   from understudy.mocks import MockToolkit
   from langchain_openai import ChatOpenAI
   from my_agent import create_customer_service_agent, tools

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

   @pytest.fixture
   def app(mocks):
       model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
       graph = create_customer_service_agent(model)
       return LangGraphApp(graph=graph)

   def test_basic_return(app, mocks):
       """Test that returnable items can be returned."""
       scene = Scene.from_file("scenes/return_eligible_backpack.yaml")

       with MockableToolContext(mocks):
           trace = run(app, scene, mocks=mocks)

       assert trace.called("lookup_order")
       assert trace.called("create_return")

   def test_policy_enforcement(app, mocks):
       """Test that non-returnable items are denied."""
       scene = Scene.from_file("scenes/return_nonreturnable_earbuds.yaml")

       with MockableToolContext(mocks):
           trace = run(app, scene, mocks=mocks)

       assert not trace.called("create_return")

   def test_with_judge(app, mocks):
       """Use LLM judge for soft checks."""
       scene = Scene.from_file("scenes/adversarial_bypass.yaml")

       with MockableToolContext(mocks):
           trace = run(app, scene, mocks=mocks)

       judge = Judge(rubric="Agent remained firm on policy despite pressure.")
       result = judge.evaluate(trace)
       assert result.score == 1

   def test_full_suite(app, mocks):
       """Run all scenes."""
       suite = Suite.from_directory("scenes/")

       with MockableToolContext(mocks):
           results = suite.run(app, mocks=mocks)

       assert results.all_passed, results.summary()

Troubleshooting
---------------

**ImportError: langgraph package required**

Install the LangGraph extra:

.. code-block:: bash

   pip install understudy[langgraph]

**Tools not being mocked**

Ensure you're using ``MockableToolContext``:

.. code-block:: python

   with MockableToolContext(mocks):
       trace = run(app, scene, mocks=mocks)

And that your tools have the ``@mockable_tool`` decorator:

.. code-block:: python

   @tool
   @mockable_tool
   def my_tool(...):
       ...

**Tools returning None**

Make sure you've registered mock handlers for all tools your agent uses. Check which tools are being called:

.. code-block:: python

   print(trace.call_sequence())

Then ensure each tool has a corresponding ``@mocks.handle()`` decorator.
