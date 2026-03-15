#!/usr/bin/env python
"""Standalone demo script for understudy.

Run with:
    cd example/adk
    python run_simulation.py

Requires GOOGLE_API_KEY for the agent.
"""

import sys
from pathlib import Path

from customer_service_agent import customer_service_agent

from understudy import Scene, check, run
from understudy.adk import ADKApp
from understudy.mocks import MockToolkit, ToolError
from understudy.storage import RunStorage


def create_mocks() -> MockToolkit:
    """Create mock handlers for the customer service agent tools."""
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
    app = ADKApp(agent=customer_service_agent)
    mocks = create_mocks()
    storage = RunStorage()

    scenes_dir = Path(__file__).parent.parent / "scenes"
    scene_files = list(scenes_dir.glob("*.yaml"))

    if not scene_files:
        print("No scene files found in scenes/")
        sys.exit(1)

    print(f"Running {len(scene_files)} scenes...\n")

    all_passed = True
    for scene_file in sorted(scene_files):
        scene = Scene.from_file(scene_file)
        print(f"=== {scene.id} ===")
        print(f"Starting: {scene.starting_prompt}")

        trace = run(app, scene, mocks=mocks)

        print(f"Turns: {trace.turn_count}")
        print(f"Tool calls: {trace.call_sequence()}")
        print(f"Terminal state: {trace.terminal_state}")

        results = check(trace, scene.expectations)
        print(results.summary())

        storage.save(trace, scene, check_result=results)

        if not results.passed:
            all_passed = False
            print("FAILED")
        else:
            print("PASSED")
        print()

    if all_passed:
        print("All scenes passed!")
        sys.exit(0)
    else:
        print("Some scenes failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
