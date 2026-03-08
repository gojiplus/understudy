LLM Judges
==========

For subjective qualities that resist deterministic checks, understudy
provides calibrated LLM-as-judge evaluation with sampling and majority vote.

Basic Usage
-----------

.. code-block:: python

   from understudy import Judge

   judge = Judge(
       rubric="The agent was empathetic and professional throughout.",
       samples=5,
   )
   result = judge.evaluate(trace)

   assert result.score == 1
   assert result.agreement_rate >= 0.6

The judge calls the LLM ``samples`` times and returns the majority-vote result.

JudgeResult
-----------

.. code-block:: python

   result.score          # 0 or 1 (majority vote)
   result.raw_scores     # individual sample scores [1, 1, 0, 1, 1]
   result.agreement_rate # fraction that agree with majority (0.8)
   result.unanimous      # True if all samples agreed

Pre-built Rubrics
-----------------

understudy includes common evaluation rubrics:

.. code-block:: python

   from understudy import (
       TOOL_USAGE_CORRECTNESS,
       POLICY_COMPLIANCE,
       TONE_EMPATHY,
       ADVERSARIAL_ROBUSTNESS,
       TASK_COMPLETION,
       FACTUAL_GROUNDING,
       INSTRUCTION_FOLLOWING,
   )

   judge = Judge(rubric=TONE_EMPATHY, samples=5)

**TOOL_USAGE_CORRECTNESS**
    Agent used appropriate tools with correct arguments.

**POLICY_COMPLIANCE**
    Agent adhered to stated policies, even under pressure.

**TONE_EMPATHY**
    Agent maintained professional, empathetic communication.

**ADVERSARIAL_ROBUSTNESS**
    Agent resisted manipulation and social engineering.

**TASK_COMPLETION**
    Agent achieved the objective efficiently.

**FACTUAL_GROUNDING**
    Agent's claims were supported by context (no hallucination).

**INSTRUCTION_FOLLOWING**
    Agent followed system prompt instructions.

Model Selection
---------------

By default, judges use ``claude-sonnet-4-20250514``. Change with:

.. code-block:: python

   judge = Judge(
       rubric=TONE_EMPATHY,
       model="gpt-4o",  # or any litellm-supported model
       samples=5,
   )

understudy uses litellm for provider access, supporting 100+ models.
Set the appropriate API key for your provider:

- ``ANTHROPIC_API_KEY`` for Claude
- ``OPENAI_API_KEY`` for OpenAI
- ``GOOGLE_API_KEY`` for Gemini

Agreement Rate
--------------

The agreement rate indicates confidence. Use it to detect borderline cases:

.. code-block:: python

   result = judge.evaluate(trace)

   if result.agreement_rate < 0.6:
       print("Low agreement - borderline case")
   elif result.unanimous:
       print("Clear pass/fail")

In CI, you might require both a passing score and high agreement:

.. code-block:: python

   assert result.score == 1, "Empathy check failed"
   assert result.agreement_rate >= 0.6, "Low confidence"

Combining Multiple Judges
-------------------------

Evaluate multiple dimensions:

.. code-block:: python

   empathy_judge = Judge(rubric=TONE_EMPATHY, samples=5)
   clarity_judge = Judge(rubric=POLICY_COMPLIANCE, samples=5)
   robustness_judge = Judge(rubric=ADVERSARIAL_ROBUSTNESS, samples=5)

   def test_agent_quality(app, scene):
       trace = run(app, scene)

       empathy = empathy_judge.evaluate(trace)
       clarity = clarity_judge.evaluate(trace)
       robustness = robustness_judge.evaluate(trace)

       assert empathy.score == 1
       assert clarity.score == 1
       assert robustness.score == 1
