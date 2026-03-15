"""LangGraph customer service agent for understudy.

This agent implements the same customer service logic as the ADK example
but using LangGraph instead of Google ADK.

Requires: pip install understudy[langgraph]
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from understudy.langgraph.tools import mockable_tool


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_PROMPT = """You are a customer service agent for TechShop.

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
"""


@tool
@mockable_tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID.

    Args:
        order_id: The order identifier (e.g., "ORD-10027")

    Returns:
        Order details including items, status, and delivery date.
    """
    raise NotImplementedError("Real implementation not available - use mocks")


@tool
@mockable_tool
def lookup_customer_orders(email: str) -> list[dict]:
    """Look up all orders for a customer by email address.

    Args:
        email: Customer email address

    Returns:
        List of order summaries for the customer.
    """
    raise NotImplementedError("Real implementation not available - use mocks")


@tool
@mockable_tool
def get_return_policy(category: str) -> dict:
    """Get the return policy for an item category.

    Args:
        category: Item category (e.g., "personal_audio", "outdoor_gear")

    Returns:
        Policy info including whether returns are allowed and conditions.
    """
    raise NotImplementedError("Real implementation not available - use mocks")


@tool
@mockable_tool
def create_return(order_id: str, item_sku: str, reason: str) -> dict:  # noqa: ARG001
    """Create a return request for an item.

    Args:
        order_id: The order identifier
        item_sku: SKU of the item to return
        reason: Customer's reason for return

    Returns:
        Return ID and shipping label URL.
    """
    raise NotImplementedError("Real implementation not available - use mocks")


@tool
@mockable_tool
def escalate_to_human(reason: str) -> dict:  # noqa: ARG001
    """Escalate the conversation to a human agent.

    Args:
        reason: Summary of why escalation is needed

    Returns:
        Escalation confirmation with ticket ID.
    """
    raise NotImplementedError("Real implementation not available - use mocks")


tools = [lookup_order, lookup_customer_orders, get_return_policy, create_return, escalate_to_human]


def create_customer_service_agent(model):
    """Create the LangGraph customer service agent.

    Args:
        model: A LangChain chat model with tools bound (e.g., ChatOpenAI(...).bind_tools(tools))

    Returns:
        Compiled LangGraph agent.
    """

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    from understudy import MockToolkit, Scene, run
    from understudy.langgraph import LangGraphApp
    from understudy.langgraph.tools import MockableToolContext

    load_dotenv()

    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
    compiled_graph = create_customer_service_agent(model)

    mocks = MockToolkit()

    @mocks.handle("lookup_order")
    def mock_lookup(order_id: str) -> dict:
        return {
            "order_id": order_id,
            "status": "delivered",
            "items": [
                {
                    "name": "Wireless Earbuds Pro",
                    "sku": "WE-500",
                    "category": "personal_audio",
                    "price": 249.99,
                }
            ],
            "date": "2025-02-15",
        }

    @mocks.handle("lookup_customer_orders")
    def mock_customer_orders(email: str) -> list[dict]:
        return [
            {
                "order_id": "ORD-10027",
                "status": "delivered",
                "items": [{"name": "Wireless Earbuds Pro", "sku": "WE-500"}],
            }
        ]

    @mocks.handle("get_return_policy")
    def mock_policy(category: str) -> dict:
        non_returnable = ["personal_audio", "perishables", "final_sale"]
        return {
            "category": category,
            "returnable": category not in non_returnable,
            "return_window_days": 30,
            "non_returnable_categories": non_returnable,
        }

    @mocks.handle("create_return")
    def mock_create_return(order_id: str, item_sku: str, reason: str) -> dict:
        return {
            "return_id": "RET-001",
            "order_id": order_id,
            "item_sku": item_sku,
            "status": "created",
        }

    @mocks.handle("escalate_to_human")
    def mock_escalate(reason: str) -> dict:
        return {"status": "escalated", "reason": reason, "ticket_id": "ESC-001"}

    from understudy.models import Expectations, Persona

    scene = Scene(
        id="return_nonreturnable",
        starting_prompt="I want to return my earbuds from order ORD-10027",
        conversation_plan="Provide order number, accept denial gracefully",
        persona=Persona.from_preset("cooperative"),
        expectations=Expectations(
            required_tools=["lookup_order", "get_return_policy"],
            forbidden_tools=["create_return"],
        ),
    )

    with MockableToolContext(mocks):
        app = LangGraphApp(graph=compiled_graph)
        trace = run(app, scene, mocks=mocks)

    print(f"Tool calls: {trace.call_sequence()}")
    print(f"Called lookup_order: {trace.called('lookup_order')}")
    print(f"Called create_return: {trace.called('create_return')}")
