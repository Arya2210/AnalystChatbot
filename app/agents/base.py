"""
Base Agent Module for LangGraph and Gemini Integration

Defines the foundational abstract BaseAgent class that handles automated
model instantiation, prompt loading, and unified state logging.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import get_settings
from app.models.context import GraphState


class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized system agents within the LangGraph network.
    
    Provides standardized hooks for accessing global configurations, instantiating
    the Google Gemini LLM interface, and processing graph transaction footprints.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Instantiate the native LangChain Gemini link using the consolidated app configurations
        params = {
            "model": "gemini-3.5-flash",
            "vertexai": True,
            "project": "project-c7187e90-0de0-4c68-b5d",
            "location": "global",
            "temperature": 0.3,
            "top_p": self.settings.gemini.top_p,
            "timeout": self.settings.gemini.timeout,
        }
        if self.settings.gemini.api_key:
            params["google_api_key"] = self.settings.gemini.api_key
        self.llm = ChatGoogleGenerativeAI(**params)

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The structural key string identifying this specific agent node 
        within the LangGraph StateGraph pipeline.
        """
        pass

    @abstractmethod
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        The entrypoint execution node interface required by LangGraph.
        
        Args:
            state: The current global GraphState memory channels dictionary.
                   
        Returns:
            Dict[str, Any]: Key-value mutations to be merged/reduced back into 
                            the state channels.
        """
        pass

    def handle_error(self, error: Exception, context_query: str) -> Dict[str, Any]:
        """
        Unified utility function to log issues and return an organized 
        fallback payload to prevent the state-graph execution thread from crashing.
        """
        error_msg = f"Exception caught in agent '{self.name}': {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        # Safely fall back to the error state format defined in response.py
        from app.models.response import AgentResponse
        fallback = AgentResponse.error_response(
            query_type=self.name.upper(),
            query=context_query,
            error=str(error)
        )
        
        return {
            "final_response": fallback
        }