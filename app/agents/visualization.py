"""
Visualization Agent Module for LangGraph

Analyzes query results available in the graph state context and structuralizes 
declarative chart configurations for frontend rendering engines.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.models.context import GraphState
from app.models.response import AgentResponse
from app.prompts.prompt_manager import PromptManager


class ChartSpecification(BaseModel):
    """Structured response model for declarative visualization metadata selection."""
    
    chart_type: str = Field(
        description="The optimal visual display form factor (e.g., 'bar', 'line', 'pie', 'scatter')."
    )
    x_axis_column: str = Field(
        description="The precise column key string string representing the independent/dimensional variable data layout."
    )
    y_axis_columns: List[str] = Field(
        description="The metric or aggregated column key string strings containing numerical values for rendering."
    )
    chart_title: str = Field(
        description="A business-focused, clean contextual title descriptive of the analytical output visual layout."
    )
    reasoning: str = Field(
        description="Step-by-step analytical logic justifying why this visualization blueprint was selected."
    )


class VisualizationAgent(BaseAgent):
    """
    Transforms tabular multi-channel dataset arrays into concrete interactive 
    chart schemas tailored for frontend data presentation layout rendering engine tracks.
    """

    @property
    def name(self) -> str:
        return "visualization_agent"

    def __init__(self) -> None:
        super().__init__()
        # Bind Gemini to structural output configuration specifications out-of-the-box
        self.structured_llm = self.llm.with_structured_output(ChartSpecification)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Main execution node interface executed natively by LangGraph engine threads.
        """
        current_query = state.get("current_query", "").strip()
        query_results: Optional[List[Dict[str, Any]]] = state.get("extracted_data")

        self.logger.info(f"Processing Visualization workflow node for query: '{current_query}'")

        # Guardrail check: Ensure there is data to actually plot
        if not query_results:
            self.logger.warning("No tabular query data discovered inside graph state space channels.")
            from app.models.response import Text2SQLResponse
            fallback = Text2SQLResponse(
                success=False,
                query=current_query,
                message="Cannot generate a visual representation because no underlying data context was found.",
                error="Missing dataset metrics dependency parameters inside active GraphState tracking.",
            )
            return {"final_response": fallback}

        try:
            # 1. Fetch clean native LangChain chat layout tracking parameters
            prompt_template = PromptManager.get_visualization_prompt(examples=True)

            # 2. Extract column structural layout blueprints from the first record row sample
            sample_keys = list(query_results[0].keys()) if len(query_results) > 0 else []
            data_context_summary = f"Available record rows count: {len(query_results)}. Table Field Keys structure: {sample_keys}"

            # 3. Request Gemini to determine chart layout framing details
            chart_chain = prompt_template | self.structured_llm
            chart_config: ChartSpecification = chart_chain.invoke({
                "query": current_query,
                "metadata": data_context_summary,
                "messages": state.get("messages", [])
            })

            self.logger.info(
                f"Successfully calculated chart parameters: [{chart_config.chart_type}] "
                f"mapping X: '{chart_config.x_axis_column}'"
            )

            # 4. Formulate response payload wrapping data arrays with chart configurations
            from app.models.response import VisualizationResponse  # Adjust to match your module file layout
            final_output = VisualizationResponse(
                success=True,
                query=current_query,
                message=f"Successfully arranged analytical display layout: {chart_config.chart_title}",
                chart_type=chart_config.chart_type,
                title=chart_config.chart_title,
                x_axis=chart_config.x_axis_column,
                y_axes=chart_config.y_axis_columns,
                data_payload=query_results,  # Merges the database records with metadata properties
                explanation=chart_config.reasoning
            )

            return {
                "final_response": final_output
            }

        except Exception as e:
            return self.handle_error(e, context_query=current_query)