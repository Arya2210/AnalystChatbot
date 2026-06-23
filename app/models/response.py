"""
Response Models for LangGraph Agents

This module contains structured output Pydantic schemas used by all specialized 
agents in the graph. These ensure that regardless of which agent executes, 
the application layer receives a standardized payload.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Base response model for all LangGraph agents in the system."""

    success: bool = Field(
        default=True, description="Whether the agent node operation was successful"
    )
    query_type: str = Field(description="The classification type of query processed (e.g., Chat, Text2SQL, Visualization)")
    query: str = Field(description="The user's original input string text")
    message: str = Field(
        description="A clear, human-readable summary message describing the outcome"
    )
    error: Optional[str] = Field(
        default=None, description="Detailed error logs or trace string if the operation failed"
    )
    needs_clarification: bool = Field(
        default=False,
        description="Flag specifying whether clarification is required from the client state",
    )
    clarification_question: Optional[str] = Field(
        default=None, description="The targeted question string to prompt back to the user"
    )

    @classmethod
    def error_response(
        cls, query_type: str, query: str, error: str
    ) -> "AgentResponse":
        """Helper factory method to instantly generate a standardized graph error state."""
        return cls(
            success=False,
            query_type=query_type,
            query=query,
            message=f"An error occurred during processing.",
            error=error,
        )

    @classmethod
    def clarification_response(
        cls, query_type: str, query: str, explanation: str, question: str
    ) -> "AgentResponse":
        """Helper factory method to seamlessly interrupt or prompt user clarification loops."""
        return cls(
            success=True,  # Set to true because handling ambiguity gracefully is an operational success
            query_type=query_type,
            query=query,
            message=explanation,
            needs_clarification=True,
            clarification_question=question,
        )


class Text2SQLResponse(AgentResponse):
    """Response validation engine model specific to the Text2SQL execution node."""

    query_type: str = "Text2SQL"
    explanation: Optional[str] = Field(
        default=None, description="Natural language reasoning or description of the query logic"
    )
    sql_query: Optional[str] = Field(
        default=None, description="The pristine generated raw PostgreSQL query text string"
    )
    query_results: Optional[dict[str, Any]] = Field(
        default=None,
        description="The tabular dict query result containing metadata and rows fetched from the database layer",
    )


class ChatResponse(AgentResponse):
    """Conversational output validation engine specific to the standard Chat agent node."""

    query_type: str = "Chat"
    answer: str = Field(description="The structural deep-dive conversational response text")


class VisualizationResponse(AgentResponse):
    """Response schema mapping specifically for data shaping and interactive Plotly configurations."""

    query_type: str = "Visualization"
    fig_json: Optional[str] = Field(
        default=None, description="The structural string JSON representation of the generated Plotly graph object"
    )
    chart_type: Optional[str] = Field(
        default=None, description="The optimal visual display form factor (e.g., 'bar', 'line', 'pie')"
    )
    title: Optional[str] = Field(
        default=None, description="A business-focused, clean contextual title descriptive of the analytical output"
    )
    x_axis: Optional[str] = Field(
        default=None, description="The precise column key string representing the independent variable"
    )
    y_axes: Optional[list[str]] = Field(
        default=None, description="The metric or aggregated column key strings containing numerical values"
    )
    data_payload: Optional[list[dict[str, Any]]] = Field(
        default=None, description="The tabular array record list fetched straight from the database layer"
    )
    explanation: Optional[str] = Field(
        default=None, description="Data scientist assessment reasoning"
    )
