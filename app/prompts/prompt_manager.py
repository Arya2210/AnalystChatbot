"""
Prompt Manager Module - Pure LangChain Native

Handles loading, management, and variable compilation of native LangChain 
ChatPromptTemplates for the multi-agent system, removing all Jinja2 dependencies.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate


class PromptManager:
    """
    Manages structured prompt templates using LangChain native formatting.
    """
    
    # Resolves to the templates folder relative to this file's path
    _template_dir: Path = Path(__file__).parent / "templates"
#   load the give filename file
    @classmethod
    def _load_raw_template(cls, file_name: str) -> str:
        """Helper to read raw text template files from disk."""
        file_path = cls._template_dir / file_name
        if not file_path.exists():
            # Fallback to the version without examples if the examples file is missing
            fallback_name = file_name.replace("_with_examples", "")
            fallback_path = cls._template_dir / fallback_name
            if fallback_path.exists():
                file_path = fallback_path
            else:
                raise FileNotFoundError(f"Template file not found at: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @classmethod
    def _build_chat_prompt(cls, system_template_name: str, user_template_name: Optional[str] = None) -> ChatPromptTemplate:
        """
        Assembles a consistent ChatPromptTemplate array structure.
        Automatically attaches a messages placeholder channel to support LangGraph history.
        """
        messages = [("system", cls._load_raw_template(system_template_name))]
        
        # If a specific user query format template is provided, inject it before the history track
        if user_template_name:
            messages.append(("human", cls._load_raw_template(user_template_name)))
            
        messages.append(("placeholder", "{messages}"))
        return ChatPromptTemplate.from_messages(messages)

    # ------ CHAT PROMPTS ------
    @classmethod
    def get_chat_prompt(cls, db_info: Optional[str] = None) -> ChatPromptTemplate:
        """Generates the system prompt structure for general chat interactions."""
        # Map to standard text files instead of .j2 extensions
        if db_info:
            prompt = cls._build_chat_prompt("chat_system_with_db.txt")
            return prompt.partial(db_info=db_info)
        
        return cls._build_chat_prompt("chat_system.txt")

    # ------ ROUTER PROMPTS ------
    @classmethod
    def get_router_prompt(cls, examples: bool = True) -> ChatPromptTemplate:
        """Generates the routing template to categorize user intents."""
        system_file = "router_system_with_examples.txt" if examples else "router_system.txt"
        return cls._build_chat_prompt(system_file, "router_user.txt")

    # ------ TEXT2SQL PROMPTS ------
    @classmethod
    def get_text2sql_generation_prompt(cls, additional_info: Optional[str] = None) -> ChatPromptTemplate:
        """Generates the compilation template for shifting Natural Language to raw SQL."""
        prompt = cls._build_chat_prompt("text2sql_generation_system.txt", "text2sql_generation_user.txt")
        return prompt.partial(additional_info=additional_info or "No additional instructions.")

    @classmethod
    def get_text2sql_verification_prompt(cls) -> ChatPromptTemplate:
        """Generates the validation prompt protecting against syntax errors or injection anomalies."""
        return cls._build_chat_prompt("text2sql_verify.txt", "text2sql_verify_user.txt")

    # ------ VISUALIZATION PROMPTS ------
    @classmethod
    def get_visualization_prompt(cls, examples: bool = True) -> ChatPromptTemplate:
        """Generates schema configuration templates for generating interactive plots."""
        system_file = "modify_query_system_with_examples.txt" if examples else "modify_query_system.txt"
        return cls._build_chat_prompt(system_file)