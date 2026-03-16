#!/usr/bin/env python
"""Standalone simulation runner for LangGraph customer service agent.

Runs all scenes against the LangGraph agent and generates an HTML report.

Usage:
    pip install understudy[langgraph,reports]
    export OPENAI_API_KEY=your-key
    cd example/langgraph
    python run_simulation.py

Output:
    - Console shows pass/fail for each scene
    - langgraph_report.html generated in current directory
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from customer_service_agent import create_customer_service_agent, tools  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from understudy import Scene, check, run  # noqa: E402
from understudy.langgraph import LangGraphApp  # noqa: E402
from understudy.langgraph.tools import MockableToolContext  # noqa: E402
from understudy.mocks import MockToolkit, ToolError  # noqa: E402
from understudy.reports import ReportGenerator  # noqa: E402
from understudy.storage import RunStorage  # noqa: E402


def create_mocks() -> MockToolkit:
    """Create mock handlers matching the scene contexts."""
    toolkit = MockToolkit()

    orders = {
        "ORD-10031": {
            "order_id": "ORD-10031",
            "items": [
                {
                    "name": "Hiking Backpack",
                    "sku": "HB-220",
                    "category": "outdoor_gear",
                    "price": 129.99,
                }
            ],
            "date": "2025-02-28",
            "status": "delivered",
        },
        "ORD-10027": {
            "order_id": "ORD-10027",
            "items": [
                {
                    "name": "Wireless Earbuds Pro",
                    "sku": "WE-500",
                    "category": "personal_audio",
                    "price": 249.99,
                }
            ],
            "date": "2025-02-15",
            "status": "delivered",
        },
    }

    non_returnable_categories = ["personal_audio", "perishables", "final_sale"]

    @toolkit.handle("lookup_order")
    def lookup_order(order_id: str) -> dict:
        if order_id in orders:
            return orders[order_id]
        raise ToolError(f"Order {order_id} not found")

    @toolkit.handle("lookup_customer_orders")
    def lookup_customer_orders(email: str) -> list[dict]:
        return list(orders.values())

    @toolkit.handle("get_return_policy")
    def get_return_policy(category: str) -> dict:
        return {
            "category": category,
            "returnable": category not in non_returnable_categories,
            "return_window_days": 30,
            "non_returnable_categories": non_returnable_categories,
        }

    @toolkit.handle("create_return")
    def create_return(order_id: str, item_sku: str, reason: str) -> dict:
        return {
            "return_id": "RET-001",
            "order_id": order_id,
            "item_sku": item_sku,
            "status": "created",
        }

    @toolkit.handle("escalate_to_human")
    def escalate_to_human(reason: str) -> dict:
        return {"status": "escalated", "reason": reason, "ticket_id": "ESC-001"}

    return toolkit


def main():
    scenes_dir = Path(__file__).parent.parent / "scenes"

    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    graph = create_customer_service_agent(model)
    app = LangGraphApp(graph=graph)

    mocks = create_mocks()
    storage = RunStorage(path=".understudy/langgraph_runs")

    storage.clear()

    scene_files = sorted(scenes_dir.glob("*.yaml"))
    print(f"Running {len(scene_files)} scenes...")
    print("-" * 60)

    results_summary = []

    for scene_file in scene_files:
        scene = Scene.from_file(scene_file)
        print(f"\n{scene.id}:")

        with MockableToolContext(mocks):
            trace = run(app, scene, mocks=mocks)

        check_result = check(trace, scene.expectations)

        storage.save(
            trace,
            scene,
            check_result=check_result,
            tags={"framework": "langgraph", "model": "gpt-4o-mini"},
        )

        status = "PASS" if check_result.passed else "FAIL"
        results_summary.append((scene.id, check_result.passed))

        print(f"  Status: {status}")
        print(f"  Terminal state: {trace.terminal_state}")
        print(f"  Tools called: {', '.join(trace.call_sequence())}")

        if not check_result.passed:
            print("  Failed checks:")
            for item in check_result.checks:
                if not item.passed:
                    print(f"    - {item.label}: {item.detail}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p in results_summary if p)
    total = len(results_summary)
    print(f"Passed: {passed}/{total}")

    for scene_id, did_pass in results_summary:
        status = "PASS" if did_pass else "FAIL"
        print(f"  [{status}] {scene_id}")

    report_dir = Path("report")
    generator = ReportGenerator(storage)
    generator.generate_static_report(report_dir)
    print(f"\nReport generated: {report_dir.absolute()}/index.html")


if __name__ == "__main__":
    main()
