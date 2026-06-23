"""
Chat Agent Module for LangGraph

Handles general conversational interactions, greetings, and analytical small talk
by evaluating state conversation message threads natively through LangChain.
"""

from typing import Any, Dict
from app.agents.base import BaseAgent
from app.models.context import GraphState
from app.models.response import ChatResponse
from app.prompts.prompt_manager import PromptManager
from app.database.connector import DatabaseConnector


class ChatAgent(BaseAgent):
    """
    Handles unstructured conversational queries that do not target 
    underlying database queries or visual presentation pipelines.
    """

    def __init__(self) -> None:
        super().__init__()
        self.connector = DatabaseConnector()

    @property
    def name(self) -> str:
        return "chat_agent"

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Main execution node interface executed natively by LangGraph engine threads.
        """
        current_query = state.get("current_query", "").strip()
        self.logger.info(f"Processing conversational node sequence for query: '{current_query}'")

        try:
            # 1. Fetch the raw template from PromptManager.
            # Retrieve schema context dynamically to allow the agent to answer questions about the database.
            db_info = None
            try:
                db_info = str(self.connector.get_text2sql_context())
            except Exception as conn_err:
                self.logger.warning(f"Could not retrieve database schema context for ChatAgent: {conn_err}")

            prompt_template = PromptManager.get_chat_prompt(db_info=db_info)

            # 2. Build your processing pipeline
            chat_chain = prompt_template | self.llm

            # 3. Invoke by sending the entire conversation history thread.
            # LangChain injects this dynamically straight into the template's placeholder.
            response = chat_chain.invoke({
                "messages": state.get("messages", [])
            })

            # 4. Package structural response values cleanly
            content = response.content
            if isinstance(content, str):
                text_answer = content
            elif isinstance(content, list):
                text_answer = "\n".join(
                    [block if isinstance(block, str) else block.get("text", "") 
                     for block in content if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")]
                )
            else:
                text_answer = str(content)

            final_output = ChatResponse(
                query=current_query,
                message="Conversation thread evaluated successfully.",
                answer=text_answer,
            )

            return {
                "final_response": final_output,
                # Append Gemini's response directly to your history tracking list
                "messages": [response]
            }

        except Exception as e:
            return self.handle_error(e, context_query=current_query)