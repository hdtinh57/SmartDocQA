from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from core.config import settings
import sys

class LLMService:
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        if self.provider == "gemini":
            if not settings.gemini_api_key:
                print("Warning: GEMINI_API_KEY is not set in .env")
            # Using Gemini Developer API configuration
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=settings.gemini_api_key,
                temperature=0.3,
                max_retries=2
            )
        else:
            # Fallback to local Ollama instance
            try:
                from langchain_community.chat_models import ChatOllama
                self.llm = ChatOllama(model="gemma2", temperature=0.3)
            except ImportError:
                print("Warning: langchain_community not installed. Cannot use Ollama.")
                self.llm = None
                
    def get_llm(self):
        """Return the Langchain LLM object for use in chains."""
        return self.llm
        
    def generate_response(self, system_prompt: str, user_query: str) -> str:
        """Simple direct QA without Langchain pipeline."""
        if not self.llm:
            return "Error: LLM not initialized."
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error during generation: {e}"
