"""pytest fixtures for understudy example tests.

This file demonstrates how to set up fixtures for testing ADK agents.
"""

from pathlib import Path

import pytest
from customer_service_agent import customer_service_agent

from understudy.adk import ADKApp
from understudy.mocks import MockToolkit, ToolError


@pytest.fixture
def app():
    """Create an ADKApp wrapping the customer service agent."""
    return ADKApp(agent=customer_service_agent)


@pytest.fixture
def scenes_dir():
    """Path to the scenes directory."""
    return Path(__file__).parent / "scenes"


@pytest.fixture
def mocks():
    """Create mock handlers for the customer service agent tools.

    These mocks simulate the backend services that the agent's tools would
    normally call. The mock data matches what's defined in the scene files.
    """
    toolkit = MockToolkit()

    # Mock order data matching scene contexts
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
