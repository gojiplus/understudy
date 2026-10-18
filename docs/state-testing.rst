State injection and next-turn tests
===================================

Use ``run_turn`` to test one user input and all agent work caused by it. It creates
an isolated app, restores or injects state, runs the input, and closes the app.
The returned trace excludes setup history, so an earlier tool call cannot satisfy
an assertion about the new exchange. No user simulator is called when ``message``
is supplied.

.. code-block:: python

   from understudy import Checkpoint, run_turn
   from understudy.adk import ADKApp
   from my_agent import build_agent

   checkpoint = Checkpoint.load("fixtures/after-turn-4.json")
   result = run_turn(
       app_factory=lambda: ADKApp(build_agent(), app_id="support-v2"),
       checkpoint=checkpoint,
       state_patch={"refund_approved": False},
       message="Please issue the refund now.",
   )
   assert not result.trace.called("issue_refund")
   assert result.state_after["refund_approved"] is False

Starting points
---------------

* ``fixture=StateFixture(state=..., history=..., tool_state=...)`` creates synthetic
  state and completed user/agent exchanges. This makes no claim to reconstruct a
  past runtime. History includes tool calls and their recorded results.
* ``checkpoint=...`` restores a captured completed boundary without executing
  prior model or tool calls. Optional ``state_patch`` creates a modified fork.
* ``replay_to(app_factory, trace, after_turn=4)`` actually executes the first four
  recorded inputs and returns a checkpoint of that new execution. By default it
  stops if response text or tool names, arguments, or results diverge. Set
  ``require_match=False`` deliberately to accept divergence.

``after_turn=4`` means four complete user-input/agent-response exchanges, including
subagent and tool activity. The existing ``Trace.turn_count`` counts individual
messages; it is not the checkpoint boundary counter.

Capture a checkpoint with ``run_turn(..., capture_checkpoint=True)`` and save
``result.checkpoint`` with its ``save(path)`` method. Repeated calls can capture
each completed exchange. ``replay_to(..., after_turn=0)`` captures a fresh runtime.
Checkpoint files include the adapter, exact framework version, application ID,
managed runtime state, history, and mock-service state. Give ``app_id`` a value
that changes when your application's state schema or graph changes. Restoring a
mismatched adapter, framework version, or application ID raises an error.

Checkpoint files are trusted test fixtures and may contain conversation data.
They do not capture external databases, provider-side memory, credentials, or
model randomness. Restore external systems through test fixtures or mock-service
state. ADK artifact storage is not included; capture rejects sessions that have
written artifacts or used long-running tools, and rejects unresolved tool calls.
LangGraph capture rejects unfinished nodes and interrupts.
Resuming inside a suspended execution is outside this completed-turn API.

To generate one user message instead of providing it, pass
``next_message=lambda history: simulator(history)``. The callback runs exactly
once and receives a copy of the prior history. Supply either ``message`` or
``next_message``, never both.

State and mocks
---------------

``result.state_before``, ``result.state_after``, and ``result.state_changes`` expose
the new exchange's state effects. ADK patches update top-level managed state keys.
LangGraph patches use its native reducers: an append reducer appends, and a message
reducer applies its message-ID rules. A patch does not universally replace values.

Use ``mocks_factory`` to create a fresh toolkit and handlers for every test. Store
mock-service data in ``toolkit.state``; checkpoint capture and restoration include
that dictionary. Handlers should read it at invocation time rather than closing
over separate mutable data. Independent factories are also required for parallel
suite and batch runs; pass ``mocks_factory`` to create per-run mocks. The ``run``
and ``simulate`` CLI commands accept a factory via ``--app`` and create a fresh
toolkit from ``--mocks`` for every run. Do not return a shared app, graph, or
toolkit from a factory.

.. code-block:: python

   from understudy import MockToolkit

   def build_mocks():
       toolkit = MockToolkit(unmatched="error", state={"balance": 100})

       @toolkit.handle("balance", agent_path=("support", "billing"))
       def balance():
           return {"balance": toolkit.state["balance"]}

       return toolkit

Agent paths scope handlers to a subtree; the longest matching prefix wins, then
an unscoped handler. ``unmatched="error"`` rejects unregistered tools before
execution. The default ``"allow"`` permits the real tool. Returning a mock result
must bypass the real tool and feed that result back to the agent.

Framework support
-----------------

ADK
~~~

``ADKApp`` installs an app plugin covering child agents, without replacing their
callbacks. Native invocation paths distinguish repeated and parallel executions.
Checkpoints restore managed session state and native events into a fresh in-memory
session. ``app_id`` defaults to the root agent name; explicit revision IDs are
recommended. Framework-owned state is captured; external services are not.

LangGraph
~~~~~~~~~

Compile the graph with a checkpointer, such as ``InMemorySaver``. Checkpoints
export native channel data and versions into a fresh thread using the framework's
JSON/msgpack serializer. Conversation messages remain part of that state. Graph
completion ends the current exchange, not the entire conversation. Use an explicit
``terminal_state`` state key to signal conversation completion.

Nested graph outputs and callback run IDs preserve node identity. Graph node paths
represent the framework's nodes; they are not inferred human agent identities.
Tool names and results are collected from message events. Decorate tools with
``understudy.langgraph.tools.mockable_tool`` and declare
``LangGraphApp(graph, tool_interception=True)`` when all tools are instrumented.
Ordinary Python calls hidden inside a node are not automatically intercepted.
Mocks use execution-local context, including framework worker threads.

LangChain
~~~~~~~~~

Install ``understudy[langchain]`` and use
``LangChainApp(model, tools, app_id="support-v2", **create_agent_options)``.
It builds a native ``create_agent`` graph with tool middleware and an in-memory
checkpointer. User middleware is retained. Both synchronous and asynchronous tool
middleware are implemented. For nested agents, build each agent with this adapter
and compose their ``graph`` objects, or install equivalent interception middleware
on every nested agent. An opaque function that creates an uninstrumented agent
cannot automatically gain tool interception.

Remote and custom adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~

HTTP/Agent Engine adapters expose next-turn traces but do not promise local tool
interception or portable state restoration. Requests for unsupported capabilities
raise before execution. A generic remote service needs its own state protocol.

Custom conversational adapters declare ``AdapterCapabilities`` and implement
``start``, ``send``, and ``stop``. State-capable adapters implement ``get_state``,
``inject``, and ``patch_state``. Checkpoint-capable adapters also implement
``checkpoint``, ``validate_checkpoint``, and ``restore``. Capabilities default to
unsupported. Tests cannot pass negative tool/agent assertions when those
observations are unavailable. The shared contract tests in ``tests/test_execution.py``
provide an executable example.

Custom autonomous ``AgenticApp`` implementations accepting mocks must declare
``supports_tool_mocking = True`` and accept ``mocks`` in ``start``. They must consult
the toolkit before executing actions and use the returned observation in their
own decision loop. The runner no longer replaces observations after execution.

pytest and CLI
--------------

Define an ``app_factory`` fixture and use the installed ``turn_runner`` fixture:

.. code-block:: python

   def test_no_refund(turn_runner, checkpoint):
       result = turn_runner("Refund this", checkpoint=checkpoint)
       assert not result.trace.called("issue_refund")

The CLI takes JSON fixture, patch, and expectation files:

.. code-block:: console

   understudy replay-until --factory my_agent:build_app --trace recorded.json --after-turn 4 --output after4.json
   understudy run-turn --factory my_agent:build_app --checkpoint after4.json --message "Refund this" --expectations expected.json --output turn5.json --checkpoint-output after5.json --runs .understudy/runs

``--fixture`` selects a synthetic fixture instead of ``--checkpoint``.
``--state-patch`` modifies state, and ``--mocks-factory`` supplies isolated mocks.
``--runs`` also saves the new exchange for the existing HTML report command.
The report identifies the starting boundary and shows native invocation IDs,
parent relationships, and state changes. A failed expectation produces a nonzero
CLI exit status and still writes the result for inspection.
