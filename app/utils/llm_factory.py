import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config.settings import get_settings

class LLMFactory:
    def __init__(self, api_key: str = None, model_name: str = None):
        settings = get_settings()
        self._api_key = api_key or (settings.gemini.api_key if settings.gemini else None) or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model_name or (settings.gemini.model if settings.gemini else None) or "gemini-2.5-flash"
        self._configured = False
        self._setup()

    def _setup(self):
        try:
            params = {
                "model": "gemini-3.5-flash",
                "vertexai": True,
                "project": "project-c7187e90-0de0-4c68-b5d",
                "location": "global",
                "temperature": 0.3,
            }
            if self._api_key:
                params["google_api_key"] = self._api_key
            self.llm = ChatGoogleGenerativeAI(**params)
            self._configured = True
        except Exception as e:
            print(f"Error configuring ChatGoogleGenerativeAI: {e}")
            self._configured = False

    def update_config(self, api_key: str, model_name: str = None):
        """Update the configuration dynamically at runtime"""
        if api_key:
            self._api_key = api_key
            if model_name:
                self._model_name = model_name
            self._setup()

    def is_configured(self) -> bool:
        return self._configured and hasattr(self, "llm")

    def generate_text(self, prompt: str, system_instruction: str = None, temperature: float = 0.1) -> str:
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. "
                "Please configure GEMINI_API_KEY in a .env file, environment variables, "
                "or input it in the Streamlit sidebar settings."
            )
        
        try:
            params = {
                "model": "gemini-3.5-flash",
                "vertexai": True,
                "project": "project-c7187e90-0de0-4c68-b5d",
                "location": "global",
                "temperature": temperature,
            }
            if self._api_key:
                params["google_api_key"] = self._api_key
            llm_to_use = ChatGoogleGenerativeAI(**params)
            
            messages = []
            if system_instruction:
                messages.append(SystemMessage(content=system_instruction))
            messages.append(HumanMessage(content=prompt))
            
            response = llm_to_use.invoke(messages)
            
            content = response.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return "\n".join([str(x) for x in content])
            else:
                return str(content)
        except Exception as e:
            # Re-raise with a clear message
            raise Exception(f"Gemini API Error: {str(e)}")

# Global LLM instance
llm_client = LLMFactory()
