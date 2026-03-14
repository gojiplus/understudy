API Reference
=============

Core Classes
------------

Scene
~~~~~

.. autoclass:: understudy.Scene
   :members:
   :undoc-members:

Persona
~~~~~~~

.. autoclass:: understudy.Persona
   :members:
   :undoc-members:

Expectations
~~~~~~~~~~~~

.. autoclass:: understudy.Expectations
   :members:
   :undoc-members:

Trace
~~~~~

.. autoclass:: understudy.Trace
   :members:
   :undoc-members:

Turn
~~~~

.. autoclass:: understudy.Turn
   :members:
   :undoc-members:

ToolCall
~~~~~~~~

.. autoclass:: understudy.ToolCall
   :members:
   :undoc-members:

Runner
------

.. autofunction:: understudy.run

.. autoclass:: understudy.AgentApp
   :members:
   :undoc-members:

Check
-----

.. autofunction:: understudy.check

.. autoclass:: understudy.CheckResult
   :members:
   :undoc-members:

Suite
-----

.. autoclass:: understudy.Suite
   :members:
   :undoc-members:

.. autoclass:: understudy.SuiteResults
   :members:
   :undoc-members:

Judges
------

.. autoclass:: understudy.Judge
   :members:
   :undoc-members:

.. autoclass:: understudy.JudgeResult
   :members:
   :undoc-members:

Rubrics
~~~~~~~

Pre-built rubrics for common evaluation dimensions:

.. py:data:: understudy.TOOL_USAGE_CORRECTNESS

   Agent used appropriate tools with correct arguments.

.. py:data:: understudy.POLICY_COMPLIANCE

   Agent adhered to stated policies, even under pressure.

.. py:data:: understudy.TONE_EMPATHY

   Agent maintained professional, empathetic communication.

.. py:data:: understudy.ADVERSARIAL_ROBUSTNESS

   Agent resisted manipulation and social engineering.

.. py:data:: understudy.TASK_COMPLETION

   Agent achieved the objective efficiently.

.. py:data:: understudy.FACTUAL_GROUNDING

   Agent's claims were supported by context (no hallucination).

.. py:data:: understudy.INSTRUCTION_FOLLOWING

   Agent followed system prompt instructions.

Storage
-------

.. autoclass:: understudy.RunStorage
   :members:
   :undoc-members:

Compare
-------

.. autofunction:: understudy.compare_runs

.. autoclass:: understudy.ComparisonResult
   :members:
   :undoc-members:

.. autoclass:: understudy.SceneComparison
   :members:
   :undoc-members:

Mocks
-----

.. autoclass:: understudy.MockToolkit
   :members:
   :undoc-members:

.. autoclass:: understudy.ToolError
   :members:
   :undoc-members:
