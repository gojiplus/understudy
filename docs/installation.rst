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

   pip install "understudy[adk]"

**LangGraph Integration**:

.. code-block:: bash

   pip install "understudy[langgraph]"

**LangChain create_agent Integration**:

.. code-block:: bash

   pip install "understudy[langchain]"

**All Extras**, including the CLI and reports:

.. code-block:: bash

   pip install "understudy[all]"

LLM judges use LiteLLM, which is included in the base installation.

**Development** from a repository checkout:

.. code-block:: bash

   uv sync --all-extras --all-groups

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
