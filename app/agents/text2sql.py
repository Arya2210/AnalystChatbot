"""
Text2SQL Agent Module for LangGraph

Translates natural language inputs into valid, structured PostgreSQL queries
using a rigorous verification and generation multi-prompt strategy.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.models.context import GraphState
from app.models.response import Text2SQLResponse
from app.prompts.prompt_manager import PromptManager
# Assuming database connector is placed here or adjust matching your tree structure
from app.database.connector import DatabaseConnector


class QueryValidationType(str, Enum):
    """Validation states for incoming analytics data queries."""
    VALID = "valid"
    REQUIRES_CLARIFICATION = "requires_clarification"
    INVALID = "invalid"


class VerificationResult(BaseModel):
    """Structured response model for checking schema capability alignment."""
    validation_status: QueryValidationType = Field(
        description="Whether the user prompt maps cleanly to available columns and metadata."
    )
    explanation: str = Field(
        description="Detailed analysis explaining the alignment or lack thereof with the database schema."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="A targeted follow-up question if information is missing or ambiguous."
    )


class SQLQuery(BaseModel):
    """Structured response model for generating secure database statements."""
    reasoning: str = Field(
        description="Step-by-step logical approach explaining how the query targets are selected."
    )
    sql_query: str = Field(
        description="The clean, executable PostgreSQL syntax string string."
    )
    explanation: str = Field(
        description="A clean, business-focused natural language explanation summary of the logic."
    )


class Text2SQLAgent(BaseAgent):
    """
    Transforms natural language requests into aggregate analytical datasets.
    Implements validation guardrails before performing infrastructure reads.
    """

    @property
    def name(self) -> str:
        return "text2sql_agent"

    def __init__(self) -> None:
        super().__init__()
        self.connector = DatabaseConnector()
        
        # Bind Gemini to structural output variants natively using LangChain
        self.verifier_llm = self.llm.with_structured_output(VerificationResult)
        self.generator_llm = self.llm.with_structured_output(SQLQuery)

    def _verify_query(self, query: str, schema_metadata: str) -> VerificationResult:
        """Evaluates whether the system has appropriate metrics to serve the request."""
        self.logger.info("Executing verification phase against schema context...")
        
        prompt_template = PromptManager.get_text2sql_verification_prompt()
        verification_chain = prompt_template | self.verifier_llm
        
        # Invoke using clean explicit keys mapped from the text files
        return verification_chain.invoke({
            "query": query,
            "metadata": schema_metadata,
            "messages": [] # Keep isolated or pull from history state if needed
        })

    def _generate_sql(self, query: str, schema_metadata: str) -> SQLQuery:
        """Compiles clean PostgreSQL strings from clear natural language inputs."""
        self.logger.info("Compiling secure PostgreSQL syntax statement...")
        
        prompt_template = PromptManager.get_text2sql_generation_prompt()
        generation_chain = prompt_template | self.generator_llm
        
        return generation_chain.invoke({
            "query": query,
            "metadata": schema_metadata,
            "messages": []
        })

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Main execution node interface called natively by LangGraph engine threads.
        """
        current_query = state.get("current_query", "").strip()
        self.logger.info(f"Processing Text2SQL workflow node for: '{current_query}'")

        try:
            # 1. Fetch live analytical database context matrix parameters
            db_context = str(self.connector.get_text2sql_context())

            # 2. Guardrail Phase: Run evaluation verification
            verification = self._verify_query(current_query, db_context)

            # Route out early if the question is invalid or unrelated
            if verification.validation_status == QueryValidationType.INVALID:
                final_output = Text2SQLResponse(
                    success=False,
                    query=current_query,
                    message=verification.explanation,
                    error="The context requested does not match available database data layouts.",
                )
                return {"final_response": final_output}

            # Handle conversational loop interruptions if parameters require clarification
            elif verification.validation_status == QueryValidationType.REQUIRES_CLARIFICATION:
                final_output = Text2SQLResponse.clarification_response(
                    query_type="Text2SQL",
                    query=current_query,
                    explanation=verification.explanation,
                    question=verification.clarification_question or "Could you please specify the timeframe or parameters?",
                )
                return {"final_response": final_output}

            # 3. Execution Phase: Generate queries for approved operations
            generated_payload = self._generate_sql(current_query, db_context)

            # 4. Production Read: Stream query directly down to connection pool
            self.logger.info(f"Dispatching generated statement to connection provider...")
            raw_records = self.connector.execute_query(generated_payload.sql_query)

            # Formulate the highly structured result instance back to state
            final_output = Text2SQLResponse(
                success=True,
                query=current_query,
                message=generated_payload.explanation,
                sql_query=generated_payload.sql_query,
                query_results=raw_records,
            )

            return {
                "final_response": final_output,
                "extracted_data": raw_records.get("rows", [])
            }

        except Exception as e:
            return self.handle_error(e, context_query=current_query)