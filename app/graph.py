"""
Graph Orchestrator Module

Assembles the multi-agent system into a stateful LangGraph workflow.
Defines execution nodes, conditional routing, and compiles the execution runtime.
"""

from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from app.models.context import GraphState
from app.agents.router import RouterAgent, QueryType
from app.agents.text2sql import Text2SQLAgent
from app.agents.visualization import VisualizationAgent
from app.agents.chat import ChatAgent


# ==========================================
# 1. Conditional Routing Logic
# ==========================================

def route_decision_condition(state: GraphState) -> Literal["text2sql_agent", "visualization_agent", "chat_agent"]:
    """
    Inspects the state modifications made by the RouterAgent 
    and determines the next node execution path.
    """
    destination = state.get("route_destination")
    
    if destination == QueryType.TEXT2SQL.value:
        return "text2sql_agent"
    elif destination == QueryType.VISUALIZATION.value:
        return "visualization_agent"
    else:
        return "chat_agent"


# ==========================================
# 2. Graph Construction & Compilation
# ==========================================

def create_agent_workflow() -> Any:
    """
    Initializes, wires, and compiles the LangGraph state machine workflow.
    """
    # Initialize StateGraph with our unified schema channel layout
    workflow = StateGraph(GraphState)

    # Instantiate Agent Nodes
    router = RouterAgent()
    text2sql = Text2SQLAgent()
    visualization = VisualizationAgent()
    chat = ChatAgent()

    # Register Nodes into the Graph Workspace
    workflow.add_node(router.name, router)
    workflow.add_node(text2sql.name, text2sql)
    workflow.add_node(visualization.name, visualization)
    workflow.add_node(chat.name, chat)

    # Establish Workflow Entrypoint
    workflow.set_entry_point(router.name)

    # Define Conditional Branching from the Router
    workflow.add_conditional_edges(
        router.name,
        route_decision_condition,
        {
            "text2sql_agent": "text2sql_agent",
            "visualization_agent": "visualization_agent",
            "chat_agent": "chat_agent"
        }
    )

    # Define Execution Terminal Paths (All roads lead back to the client response)
    workflow.add_edge("text2sql_agent", END)
    workflow.add_edge("visualization_agent", END)
    workflow.add_edge("chat_agent", END)

    # Compile the graph into an executable application state machine
    return workflow.compile()


# Global application executable instance
app = create_agent_workflow()