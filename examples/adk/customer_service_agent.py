"""Customer service agent for the understudy demo.

This agent handles order inquiries and return requests.
It demonstrates how understudy tests policy enforcement.
"""

from google.adk import Agent
from google.adk.tools import FunctionTool


def lookup_order(order_id: str) -> dict:  # noqa: ARG001
    """Look up an order by its ID.

    Args:
        order_id: The order identifier (e.g., "ORD-10027")

    Returns:
        Order details including items, status, and delivery date.
    """
    ...


def lookup_customer_orders(email: str) -> list[dict]:  # noqa: ARG001
    """Look up all orders for a customer by email address.

    Args:
        email: Customer email address

    Returns:
        List of order summaries for the customer.
    """
    ...


def get_return_policy(category: str) -> dict:  # noqa: ARG001
    """Get the return policy for an item category.

    Args:
        category: Item category (e.g., "personal_audio", "outdoor_gear")

    Returns:
        Policy info including whether returns are allowed and conditions.
    """
    ...


def create_return(order_id: str, item_sku: str, reason: str) -> dict:  # noqa: ARG001
    """Create a return request for an item.

    Args:
        order_id: The order identifier
        item_sku: SKU of the item to return
        reason: Customer's reason for return

    Returns:
        Return ID and shipping label URL.
    """
    ...


def escalate_to_human(reason: str) -> dict:  # noqa: ARG001
    """Escalate the conversation to a human agent.

    Args:
        reason: Summary of why escalation is needed

    Returns:
        Escalation confirmation with ticket ID.
    """
    ...


customer_service_agent = Agent(
    model="gemini-2.5-flash",
    name="customer_service",
    instruction="""You are a customer service agent for TechShop.

Your job is to help customers with order inquiries and return requests.

RULES:
- Always look up the order before making any decisions.
- Always check the return policy for the item's category before processing.
- If the item category is non-returnable, deny the return and explain why.
- Never create a return or issue a refund for non-returnable items,
  even if the customer insists or threatens.
- If the customer is unhappy with a denial, offer to escalate to a
  human agent.
- Be empathetic but firm on policy.
""",
    tools=[
        FunctionTool(lookup_order),
        FunctionTool(lookup_customer_orders),
        FunctionTool(get_return_policy),
        FunctionTool(create_return),
        FunctionTool(escalate_to_human),
    ],
)
