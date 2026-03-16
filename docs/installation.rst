Installation
============

Basic Installation
------------------

Install understudy from PyPI:

.. code-block:: bash

   pip install understudy

Or with uv:

.. code-block:: bash

   uv add understudy

Optional Dependencies
---------------------

understudy has several optional dependency groups:

**ADK Integration** (Google Agent Development Kit):

.. code-block:: bash

   pip install understudy[adk]

**LangGraph Integration** (LangChain/LangGraph support):

.. code-block:: bash

   pip install understudy[langgraph]

**LLM Judges** (litellm for 100+ provider support):

.. code-block:: bash

   pip install understudy[judges]

**All Extras**:

.. code-block:: bash

   pip install understudy[all]

**Development** (includes testing and documentation tools):

.. code-block:: bash

   pip install understudy[dev]

Requirements
------------

- Python 3.12 or higher
- pydantic >= 2.0
- pyyaml >= 6.0

For LLM judges, you'll need API keys for your chosen provider:

- ``ANTHROPIC_API_KEY`` for Claude models
- ``OPENAI_API_KEY`` for OpenAI models
- ``GOOGLE_API_KEY`` for Gemini models
- Or any other provider supported by litellm
