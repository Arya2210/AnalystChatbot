import os
import google.generativeai as genai
from app.config.settings import settings
# utilize langchain model interface
class LLMFactory:
    def __init__(self, api_key: str = None, model_name: str = None):
        self._api_key = api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._model_name = model_name or settings.model_name or "gemini-1.5-flash"
        self._configured = False
        self._setup()

    def _setup(self):
        if self._api_key:
            try:
                genai.configure(api_key=self._api_key)
                self._configured = True
            except Exception as e:
                print(f"Error configuring Google Generative AI: {e}")
                self._configured = False

    def update_config(self, api_key: str, model_name: str = None):
        """Update the configuration dynamically at runtime"""
        if api_key:
            self._api_key = api_key
            if model_name:
                self._model_name = model_name
            self._setup()

    def is_configured(self) -> bool:
        return self._configured and bool(self._api_key)

    def generate_text(self, prompt: str, system_instruction: str = None, temperature: float = 0.1) -> str:
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. "
                "Please configure GEMINI_API_KEY in a .env file, environment variables, "
                "or input it in the Streamlit sidebar settings."
            )
        
        try:
            config = genai.types.GenerationConfig(
                temperature=temperature
            )
            model = genai.GenerativeModel(
                model_name=self._model_name,
                generation_config=config,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Re-raise with a clear message
            raise Exception(f"Gemini API Error: {str(e)}")

# Global LLM instance
llm_client = LLMFactory()
