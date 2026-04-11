Examples
========

Customer Service Agent
----------------------

A complete example demonstrating understudy with a customer service agent
is available in the ``examples/`` directory.

The example includes:

- A customer service agent with order lookup and return handling
- Scene files for various scenarios (returns, policy enforcement, adversarial)
- Test files showing deterministic assertions and LLM judge usage

Running the Example
~~~~~~~~~~~~~~~~~~~

1. Install dependencies:

   .. code-block:: bash

      pip install understudy[all]
      # or
      uv add understudy[all]

2. Set your API key:

   .. code-block:: bash

      export ANTHROPIC_API_KEY=your-key
      # or
      export GOOGLE_API_KEY=your-key

3. Run the simulation:

   .. code-block:: bash

      cd examples
      python run_simulation.py

4. Run tests:

   .. code-block:: bash

      pytest examples/adk/test_adk_returns.py -v

Scene Files
~~~~~~~~~~~

The example includes these scenes:

**return_eligible_backpack.yaml**
    Customer returns a backpack (eligible category). Agent should process the return.

**return_nonreturnable_earbuds.yaml**
    Customer returns earbuds (non-returnable category). Agent should deny.

**adversarial_policy_bypass.yaml**
    Customer attempts social engineering to bypass return policy.

CI Integration
--------------

Example GitHub Actions workflow:

.. code-block:: yaml

   name: Agent Tests
   on: [push, pull_request]

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.12"
         - run: pip install understudy[all] pytest
         - run: pytest tests/ --junitxml=results.xml
           env:
             ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
         - uses: dorny/test-reporter@v2
           if: always()
           with:
             name: Understudy Results
             path: results.xml
             reporter: java-junit
