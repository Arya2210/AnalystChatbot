"""
Query Router Node Module for LangGraph

Analyzes user interaction sequences and maps structured intent values 
into the graph state to drive conditional engine edge routing.
"""

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.models.context import GraphState
from app.prompts.prompt_manager import PromptManager


class QueryType(str, Enum):
    """Supported specialized node destinations across the graph."""
    TEXT2SQL = "text2sql"
    VISUALIZATION = "visualization"
    CHAT = "chat"


class QueryClassification(BaseModel):
    """Pydantic model bound to Gemini for structured routing execution."""
    
    query_type: QueryType = Field(
        description="The target category most relevant to the query context."
    )
    confidence_score: float = Field(
        description="Confidence calibration score ranging between 0.0 and 1.0"
    )
    updated_query: str = Field(
        default="",
        description="An optimized re-phrasing of the user input if context history implies an incomplete question.",
    )


class RouterAgent(BaseAgent):
    """
    Graph router node responsible for evaluating transaction intent 
    and mutating state targets to govern graph branch movement.
    """

    @property
    def name(self) -> str:
        return "router_agent"

    def __init__(self) -> None:
        super().__init__()
        # Secure structural extraction capabilities natively out of Gemini
        self.structured_llm = self.llm.with_structured_output(QueryClassification)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Processes current query and history to update destination routing keys.
        """
        current_input = state.get("current_query", "").strip()
        self.logger.info(f"Classifying routing destination for incoming query: '{current_input}'")

        try:
            # 1. Fetch your clean native LangChain chat template layout
            prompt_template = PromptManager.get_router_prompt()

            # 2. Bind the template to our structured output model pipeline
            routing_chain = prompt_template | self.structured_llm

            # 3. Invoke by feeding LangGraph's native message history channels directly
            result: QueryClassification = routing_chain.invoke({
                "messages": state["messages"]
            })

            # Handle contextual query updates if Gemini provides an optimized version
            effective_query = result.updated_query if result.updated_query.strip() else current_input
            
            self.logger.info(
                f"Successfully routed to target branch [{result.query_type.value}] "
                f"with confidence {result.confidence_score}"
            )

            # Return state updates to merge back into LangGraph channel pipelines
            return {
                "route_destination": result.query_type.value,
                "current_query": effective_query
            }

        except Exception as e:
            self.logger.error(f"Routing evaluation fell over. Forcing fallback gracefully. Error: {e}")
            # Safe baseline fallback configuration to keep processing alive
            return {
                "route_destination": QueryType.CHAT.value,
                "current_query": current_input
            }