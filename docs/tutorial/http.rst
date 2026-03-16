HTTP Client
===========

Test deployed ADK agents via REST API without importing the agent code.

HTTPApp
-------

For agents deployed with a REST endpoint:

.. code-block:: python

   from understudy.http import HTTPApp
   from understudy import Scene, run

   app = HTTPApp(
       base_url="http://localhost:8080",
       app_name="customer_service",
   )

   scene = Scene.from_file("scenes/test_scenario.yaml")
   trace = run(app, scene)

   assert trace.called("lookup_order")

Configuration options:

.. code-block:: python

   app = HTTPApp(
       base_url="http://localhost:8080",
       app_name="customer_service",
       user_id="test_user",           # Custom user ID
       headers={"X-API-Key": "..."},  # Custom headers
       timeout=120.0,                 # Request timeout in seconds
   )

Agent Engine
------------

For agents deployed on Google Cloud Agent Engine:

.. code-block:: python

   from understudy.http import AgentEngineApp
   from understudy import Scene, run

   app = AgentEngineApp(
       project_id="my-gcp-project",
       location="us-central1",
       resource_id="my-agent-resource-id",
   )

   scene = Scene.from_file("scenes/test_scenario.yaml")
   trace = run(app, scene)

Uses Google Cloud default credentials. To use specific credentials:

.. code-block:: python

   from google.oauth2 import service_account

   creds = service_account.Credentials.from_service_account_file(
       "service-account.json"
   )

   app = AgentEngineApp(
       project_id="my-gcp-project",
       location="us-central1",
       resource_id="my-agent-resource-id",
       credentials=creds,
   )

Installation
------------

Install the HTTP extra:

.. code-block:: bash

   pip install understudy[http]

For Agent Engine, also install Google auth:

.. code-block:: bash

   pip install understudy[http] google-auth

Note on Mocking
---------------

When testing deployed agents via HTTP, mocks are **not injected** into the agent.
The deployed agent uses its real tool implementations.

If you need to test with mocked tool responses, either:

1. Deploy a test version of the agent with mocked backends
2. Use ``ADKApp`` with the agent code directly (recommended for unit testing)
