Assertions
==========

understudy provides deterministic assertions against the **trace** —
what actually happened, not what the agent said about what happened.

The Trace
---------

After running a scene, you get a ``Trace`` that records:

- All turns (user and agent messages)
- All tool calls with arguments and results
- The terminal state
- Timing information

.. code-block:: python

   trace = run(app, scene)

   trace.turns           # list of Turn objects
   trace.tool_calls      # flat list of all tool invocations
   trace.terminal_state  # final resolution state
   trace.turn_count      # number of turns
   trace.duration        # wall clock time

Querying Tool Calls
-------------------

Check if a tool was called:

.. code-block:: python

   assert trace.called("lookup_order")
   assert not trace.called("issue_refund")

Check with specific arguments:

.. code-block:: python

   assert trace.called("lookup_order", order_id="ORD-10027")

Get all calls to a tool:

.. code-block:: python

   calls = trace.calls_to("lookup_order")
   assert calls[0].arguments["order_id"] == "ORD-10027"

Get the sequence of tool calls:

.. code-block:: python

   assert trace.call_sequence() == ["lookup_order", "get_return_policy"]

Terminal State Assertions
-------------------------

Check the final resolution:

.. code-block:: python

   assert trace.terminal_state == "return_denied_policy"
   # or allow multiple valid outcomes
   assert trace.terminal_state in {"return_denied_policy", "escalated_to_human"}

Bulk Check
----------

Use ``check()`` to validate all expectations at once:

.. code-block:: python

   from understudy import check

   results = check(trace, scene.expectations)
   assert results.passed, f"Failed:\n{results.summary()}"

The summary shows what passed and failed:

.. code-block:: text

   ✓ required_tools: lookup_order, get_return_policy
   ✓ forbidden_tools: create_return (none called)
   ✗ terminal_state: return_created (expected: return_denied_policy)

Pytest Integration
------------------

understudy is designed to work with pytest:

.. code-block:: python

   import pytest
   from understudy import Scene, run, check

   def test_policy_enforcement(app):
       scene = Scene.from_file("scenes/return_nonreturnable.yaml")
       trace = run(app, scene)

       assert trace.called("get_return_policy")
       assert not trace.called("create_return")
       assert trace.terminal_state == "return_denied_policy"

   @pytest.mark.parametrize("scene_file", [
       "scenes/return_nonreturnable_earbuds.yaml",
       "scenes/return_nonreturnable_perishable.yaml",
   ])
   def test_denial_scenarios(app, scene_file):
       scene = Scene.from_file(scene_file)
       trace = run(app, scene)
       results = check(trace, scene.expectations)
       assert results.passed
