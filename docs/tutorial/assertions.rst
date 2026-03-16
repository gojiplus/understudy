Assertions
==========

understudy provides deterministic assertions against the **trace** —
what actually happened, not what the agent said about what happened.

The Trace
---------

After running a scene, you get a ``Trace`` that records:

- All turns (user and agent messages)
- All tool calls with arguments and results
- Run status (completed or max_turns_reached)
- Timing information

.. code-block:: python

   trace = run(app, scene)

   trace.turns           # list of Turn objects
   trace.tool_calls      # flat list of all tool invocations
   trace.terminal_state  # "completed" or "max_turns_reached"
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

Trajectory Matching
-------------------

Compare tool call sequences against expected patterns:

.. code-block:: yaml

   expectations:
     expected_trajectory: [lookup_order, get_return_policy, create_return]
     trajectory_match_mode: exact

Match modes:

- **exact**: Sequences must match exactly
- **prefix**: Expected sequence appears at the start of actual
- **contains**: Expected tools appear in order within actual (other tools allowed between)
- **subset**: All expected tools were called (any order)

Use the ``trajectory_match`` metric to evaluate:

.. code-block:: python

   expectations = Expectations(
       expected_trajectory=["lookup_order", "get_return_policy"],
       trajectory_match_mode="contains",
       metrics=["trajectory_match"],
   )
   result = check(trace, expectations)
   assert result.metrics["trajectory_match"].passed

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

   @pytest.mark.parametrize("scene_file", [
       "scenes/return_nonreturnable_earbuds.yaml",
       "scenes/return_nonreturnable_perishable.yaml",
   ])
   def test_denial_scenarios(app, scene_file):
       scene = Scene.from_file(scene_file)
       trace = run(app, scene)
       results = check(trace, scene.expectations)
       assert results.passed
