from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Smart Document Q&A System"
    env: str = "development"
    
    # API Keys
    mistral_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    
    # Qdrant Database
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str = Field(default="")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
