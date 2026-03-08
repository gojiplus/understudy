Scenes
======

A **Scene** is a conversation fixture that defines everything needed to run
a simulated conversation against your agent.

Scene Components
----------------

**id**
    A unique identifier for the scene.

**starting_prompt**
    The first message from the simulated user.

**conversation_plan**
    Natural language instructions for how the simulated user should behave.
    This guides the user's responses throughout the conversation.

**persona**
    The personality and behavioral patterns of the simulated user.

**max_turns**
    Maximum number of conversation turns before timeout.

**context**
    World state that seeds mock services and provides context to the simulator.

**expectations**
    Assertions to verify against the trace.

Creating Scenes in Python
-------------------------

.. code-block:: python

   from understudy import Scene, Persona, Expectations

   scene = Scene(
       id="order_inquiry",
       starting_prompt="Where's my order?",
       conversation_plan="""
           Goal: Get status of order ORD-12345.
           - Provide order ID when asked.
           - Cooperate with verification steps.
       """,
       persona=Persona.COOPERATIVE,
       max_turns=15,
       context={
           "orders": {
               "ORD-12345": {"status": "shipped", "eta": "2025-03-10"}
           }
       },
       expectations=Expectations(
           required_tools=["lookup_order"],
           allowed_terminal_states=["order_info_provided"],
       ),
   )

Creating Scenes in YAML
-----------------------

YAML is often more readable for scene definitions:

.. code-block:: yaml

   id: order_inquiry
   description: Customer asking about order status.

   starting_prompt: "Where's my order?"
   conversation_plan: |
     Goal: Get status of order ORD-12345.
     - Provide order ID when asked.
     - Cooperate with verification steps.

   persona: cooperative
   max_turns: 15

   context:
     orders:
       ORD-12345:
         status: shipped
         eta: "2025-03-10"

   expectations:
     required_tools:
       - lookup_order
     allowed_terminal_states:
       - order_info_provided

Load with:

.. code-block:: python

   scene = Scene.from_file("scenes/order_inquiry.yaml")

Personas
--------

Built-in presets:

- ``Persona.COOPERATIVE`` - Direct, helpful, provides info when asked
- ``Persona.FRUSTRATED_BUT_COOPERATIVE`` - Pushes back once, then cooperates
- ``Persona.ADVERSARIAL`` - Tries to social-engineer exceptions
- ``Persona.VAGUE`` - Gives incomplete info, needs follow-up
- ``Persona.IMPATIENT`` - Short answers, wants fast resolution

Custom personas:

.. code-block:: python

   persona = Persona(
       description="elderly, not tech-savvy",
       behaviors=[
           "asks for clarification on technical terms",
           "provides information slowly",
       ],
   )

In YAML:

.. code-block:: yaml

   persona:
     description: "elderly, not tech-savvy"
     behaviors:
       - asks for clarification on technical terms
       - provides information slowly
