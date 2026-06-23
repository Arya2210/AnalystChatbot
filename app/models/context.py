"""
Context Module for LangGraph State Management

Defines the centralized State tracking object used across the multi-agent graph system,
leveraging LangChain's message reducers for session history.
"""

from typing import Any, Annotated, TypedDict, Union
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.models.response import AgentResponse, Text2SQLResponse, ChatResponse, VisualizationResponse



class GraphState(TypedDict):
    """
    The blueprint for the global graph state memory channel.
    
    Any node in the graph can read from this dictionary, and returning updates
    from a node will automatically merge or reduce values into these channels.
    """
    
    # LangChain's add_messages reducer appends new messages automatically
    # instead of overwriting the historical list channel.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Tracks the most recent natural language user query string
    current_query: str
    
    # Captures intent routing classifications ('chat', 'text2sql', 'visualization')
    route_destination: str
    
    # Stores raw dataset extractions (e.g., list of dict records from PostgreSQL)
    extracted_data: list[dict[str, Any]]
    
    # Final unified structured response delivery payload
    final_response: Union[AgentResponse, Text2SQLResponse, ChatResponse, VisualizationResponse]