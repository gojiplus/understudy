understudy
==========

**Simulation and trace-based evaluation for agentic systems.**

The simulated user is an understudy standing in for a real user.
You write scenes, run rehearsals, and check the performance —
not by reading the script, but by inspecting what actually happened.

.. code-block:: bash

   pip install understudy

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   adk-integration
   tutorial/index
   api/index
   examples/index

What It Does
------------

``understudy`` is a testing layer, not a platform. It gives you:

1. **Scene files** — conversation fixtures that encode world state, user intent, persona, and expectations
2. **A simulator** — drives the user side dynamically against your running agent
3. **Trace capture** — records tool calls, arguments, state transitions, terminal resolution
4. **Deterministic assertions** — pytest-native checks against the trace, not the prose
5. **Calibrated LLM-as-judge** — for soft dimensions (tone, clarity), with sampling and majority vote
6. **A mock layer** — fake downstream services seeded from the scene's context

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
