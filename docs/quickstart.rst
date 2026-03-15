Quickstart
==========

This guide walks you through creating your first understudy test.

Creating a Scene
----------------

A Scene defines a test scenario with:

- The starting prompt
- A conversation plan for the simulated user
- World state (context)
- Expectations for what should/shouldn't happen

.. code-block:: python

   from understudy import Scene, Persona, Expectations

   scene = Scene(
       id="return_nonreturnable_item",
       starting_prompt="I want to return something I bought.",
       conversation_plan="""
           Goal: Return earbuds from order ORD-10027.
           - If asked for order ID: provide ORD-10027
           Return reason: defective.
           If denied: accept escalation.
       """,
       persona=Persona.FRUSTRATED_BUT_COOPERATIVE,
       max_turns=20,
       context={
           "orders": {
               "ORD-10027": {
                   "items": [{"name": "Earbuds", "category": "personal_audio"}],
                   "status": "delivered",
               }
           },
           "policy": {
               "non_returnable_categories": ["personal_audio"],
           },
       },
       expectations=Expectations(
           required_tools=["lookup_order", "get_return_policy"],
           forbidden_tools=["create_return"],
       ),
   )

Or load from YAML:

.. code-block:: python

   scene = Scene.from_file("scenes/return_nonreturnable_item.yaml")

Running a Rehearsal
-------------------

.. code-block:: python

   from understudy import run
   from understudy.adk import ADKApp

   app = ADKApp(agent=your_agent)
   trace = run(app, scene)

Making Assertions
-----------------

Assert against the trace (what happened), not the prose:

.. code-block:: python

   def test_policy_enforcement():
       trace = run(app, scene)

       assert trace.called("lookup_order")
       assert trace.called("get_return_policy")
       assert not trace.called("create_return")

Using LLM Judges
----------------

For subjective qualities like tone and clarity:

.. code-block:: python

   from understudy import Judge, TONE_EMPATHY

   judge = Judge(rubric=TONE_EMPATHY, samples=5)
   result = judge.evaluate(trace)

   assert result.score == 1
   assert result.agreement_rate >= 0.6

Running a Suite
---------------

Run all scenes in a directory:

.. code-block:: python

   from understudy import Suite

   suite = Suite.from_directory("scenes/")
   results = suite.run(app, parallel=4)
   results.to_junit_xml("test-results/understudy.xml")
   assert results.all_passed

Tagging and Comparing Runs
--------------------------

Tag runs for later comparison (e.g., comparing model versions):

.. code-block:: python

   from understudy import Suite, RunStorage

   storage = RunStorage()
   suite = Suite.from_directory("scenes/")

   # Tag runs with version
   suite.run(app, storage=storage, tags={"version": "v1"})
   suite.run(app, storage=storage, tags={"version": "v2"})

Then compare via CLI:

.. code-block:: bash

   # CLI comparison
   understudy compare --tag version --before v1 --after v2

   # HTML report
   understudy compare --tag version --before v1 --after v2 --html comparison.html

The comparison shows per-scene pass rates when running multiple times.
