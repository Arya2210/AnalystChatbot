"""
Text2SQL Agent Module for LangGraph

Translates natural language inputs into valid, structured SQL queries
using a stateful LangGraph subgraph with self-correcting validation loops.
"""

import re
import logging
from enum import Enum
from typing import Any, Dict, Optional, Annotated, TypedDict, Union
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.agents.base import BaseAgent
from app.models.context import GraphState
from app.models.response import Text2SQLResponse, AgentResponse
from app.prompts.prompt_manager import PromptManager
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
        description="The clean, executable SQL syntax string."
    )
    explanation: str = Field(
        description="A clean, business-focused natural language explanation summary of the logic."
    )


class Text2SQLState(TypedDict):
    """
    State blueprint for the Text-to-SQL subgraph.
    Inherits fields from GraphState and manages subgraph-specific channels.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    current_query: str
    route_destination: str
    extracted_data: list[dict[str, Any]]
    final_response: Union[AgentResponse, Text2SQLResponse]

    # Subgraph specific channels
    sql_query: str
    explanation: str
    error_message: str
    retry_count: int
    validation_status: str


class Text2SQLAgent(BaseAgent):
    """
    Transforms natural language requests into aggregate analytical datasets.
    Provides methods for the Text-to-SQL subgraph nodes and orchestrates execution.
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
        self.raw_llm = self.llm

    def verify_query_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Evaluates whether the system has appropriate metrics to serve the request."""
        query = state.get("current_query", "").strip()
        self.logger.info(f"Executing verification phase against schema context for: '{query}'")
        
        try:
            db_context = str(self.connector.get_text2sql_context())
            prompt_template = PromptManager.get_text2sql_verification_prompt()
            verification_chain = prompt_template | self.verifier_llm
            
            verification = verification_chain.invoke({
                "query": query,
                "metadata": db_context,
                "messages": []
            })

            if verification.validation_status == QueryValidationType.INVALID:
                final_output = Text2SQLResponse(
                    success=False,
                    query=query,
                    message=verification.explanation,
                    error="The context requested does not match available database data layouts.",
                )
                return {
                    "final_response": final_output,
                    "validation_status": "invalid"
                }

            elif verification.validation_status == QueryValidationType.REQUIRES_CLARIFICATION:
                final_output = Text2SQLResponse.clarification_response(
                    query_type="Text2SQL",
                    query=query,
                    explanation=verification.explanation,
                    question=verification.clarification_question or "Could you please specify the timeframe or parameters?",
                )
                return {
                    "final_response": final_output,
                    "validation_status": "requires_clarification"
                }

            return {
                "validation_status": "valid",
                "retry_count": 0
            }
        except Exception as e:
            self.logger.error(f"Error during verification: {e}", exc_info=True)
            return {
                "validation_status": "invalid",
                "final_response": self.handle_error(e, context_query=query)["final_response"]
            }

    def generate_query_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Compiles SQL query representation using schema metadata."""
        query = state.get("current_query", "").strip()
        self.logger.info("Compiling initial SQL query...")
        
        try:
            db_context = str(self.connector.get_text2sql_context())
            prompt_template = PromptManager.get_text2sql_generation_prompt()
            generation_chain = prompt_template | self.generator_llm
            
            generated = generation_chain.invoke({
                "query": query,
                "metadata": db_context,
                "messages": []
            })
            
            return {
                "sql_query": generated.sql_query,
                "explanation": generated.explanation
            }
        except Exception as e:
            self.logger.error(f"Error during generation: {e}", exc_info=True)
            return {
                "sql_query": "",
                "error_message": f"Initial query generation failed: {str(e)}",
                "validation_status": "generation_failed"
            }

    def validate_query_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Validates query for security constraints and clean syntax."""
        query = state.get("sql_query", "").strip()
        self.logger.info(f"Validating SQL query: {query}")
        
        # Clean formatting
        clean_query = query
        clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'```\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = clean_query.strip().rstrip(";")
        
        # Security constraints
        if not clean_query:
            return {
                "validation_status": "syntax_invalid",
                "error_message": "Error: Empty query generated"
            }

        if "select" not in clean_query.lower():
            return {
                "validation_status": "syntax_invalid",
                "error_message": "Error: Query must contain 'select'"
            }
            
        danger_keywords = ['drop', 'delete', 'update', 'insert', 'create', 'alter', 'truncate', 'merge', 'with']
        for keyword in danger_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', clean_query.lower()):
                return {
                    "validation_status": "safety_invalid",
                    "error_message": f"Error: Query contains dangerous keyword: {keyword}"
                }
                
        return {
            "sql_query": clean_query,
            "validation_status": "safe"
        }

    def execute_query_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Runs validated query on database."""
        query = state.get("sql_query", "").strip()
        self.logger.info(f"Executing SQL query on DB: {query}")
        
        try:
            result = self.connector.execute_query(query)
            
            if result.get("success", False):
                final_output = Text2SQLResponse(
                    success=True,
                    query=state.get("current_query", ""),
                    message=state.get("explanation", "Query executed successfully."),
                    sql_query=query,
                    query_results=result
                )
                return {
                    "final_response": final_output,
                    "extracted_data": result.get("rows", []),
                    "validation_status": "execution_success"
                }
            else:
                return {
                    "validation_status": "execution_failed",
                    "error_message": result.get("error", "Database execution error")
                }
        except Exception as e:
            return {
                "validation_status": "execution_failed",
                "error_message": f"Execution node exception: {str(e)}"
            }

    def fix_query_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Corrects SQL query using LLM and database schema context."""
        query = state.get("sql_query", "").strip()
        error_msg = state.get("error_message", "")
        question = state.get("current_query", "")
        retry_count = state.get("retry_count", 0) + 1
        
        self.logger.info(f"Attempting to fix query. Retry count: {retry_count}. Error: {error_msg}")
        
        try:
            db_context = str(self.connector.get_text2sql_context())
            
            fix_prompt = f"""You are an expert SQL engineer.
The SQL query you generated failed.
Original Question: {question}
Generated SQL Query: {query}
Error Message: {error_msg}
Database Schema: {db_context}

Analyze the error and generate a corrected SQL query.
Rules:
1. Fix the specific issue mentioned.
2. Return ONLY the raw SQL query. Do NOT include any markdown formatting like ```sql or explanation.
3. Only generate SELECT queries.
4. Ensure all column names, table names, and table relationships match the provided database schema.
"""
            response = self.raw_llm.invoke(fix_prompt)
            fixed_query = response.content.strip()
            
            # Clean any backticks that Gemini might have returned despite rules
            fixed_query = re.sub(r'```sql\s*', '', fixed_query, flags=re.IGNORECASE)
            fixed_query = re.sub(r'```\s*', '', fixed_query, flags=re.IGNORECASE)
            fixed_query = fixed_query.strip()
            
            return {
                "sql_query": fixed_query,
                "retry_count": retry_count
            }
        except Exception as e:
            return {
                "retry_count": retry_count,
                "error_message": f"Fix node exception: {str(e)}"
            }

    def fail_node(self, state: Text2SQLState) -> Dict[str, Any]:
        """Fallback fail node when retry limit is reached."""
        self.logger.error("Maximum retries reached. Returning failure response.")
        
        final_output = Text2SQLResponse(
            success=False,
            query=state.get("current_query", ""),
            message="Could not generate a valid query after multiple attempts.",
            sql_query=state.get("sql_query"),
            error=state.get("error_message", "Retry limit exceeded")
        )
        return {
            "final_response": final_output
        }

    def build_graph(self):
        """Constructs and compiles the StateGraph."""
        subgraph = StateGraph(Text2SQLState)
        
        # Add Nodes
        subgraph.add_node("verify_query", self.verify_query_node)
        subgraph.add_node("generate_query", self.generate_query_node)
        subgraph.add_node("validate_query", self.validate_query_node)
        subgraph.add_node("execute_query", self.execute_query_node)
        subgraph.add_node("fix_query", self.fix_query_node)
        subgraph.add_node("fail", self.fail_node)
        
        # Set Entry Point
        subgraph.set_entry_point("verify_query")
        
        # Define Edges
        # 1. From Verification
        subgraph.add_conditional_edges(
            "verify_query",
            lambda state: state.get("validation_status"),
            {
                "valid": "generate_query",
                "invalid": END,
                "requires_clarification": END
            }
        )
        
        # 2. From Generation
        subgraph.add_edge("generate_query", "validate_query")
        
        # 3. From Validation
        def after_validation_routing(state: Text2SQLState) -> str:
            if state.get("validation_status") == "safe":
                return "execute_query"
            elif state.get("retry_count", 0) < 3:
                return "fix_query"
            else:
                return "fail"
                
        subgraph.add_conditional_edges(
            "validate_query",
            after_validation_routing,
            {
                "execute_query": "execute_query",
                "fix_query": "fix_query",
                "fail": "fail"
            }
        )
        
        # 4. From Execution
        def after_execution_routing(state: Text2SQLState) -> str:
            if state.get("validation_status") == "execution_success":
                return END
            elif state.get("retry_count", 0) < 3:
                return "fix_query"
            else:
                return "fail"
                
        subgraph.add_conditional_edges(
            "execute_query",
            after_execution_routing,
            {
                END: END,
                "fix_query": "fix_query",
                "fail": "fail"
            }
        )
        
        # 5. From Fix Query Node back to Validation
        subgraph.add_edge("fix_query", "validate_query")
        
        # 6. From Fail Node
        subgraph.add_edge("fail", END)
        
        # Compile
        compiled_graph = subgraph.compile()
        compiled_graph.name = self.name
        return compiled_graph

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Fallback callable node representation."""
        compiled_graph = self.build_graph()
        return compiled_graph.invoke(state)