from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env", extra="ignore")
    
    APP_ENV: Literal["prototype", "development", "production"]
    
    HF_TOKEN: str
    MONGO_USER: str
    MONGO_PASSWORD: str
    MONGO_URI: str
    MONGO_DB: str
    OPENROUTER_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    SENTRY_DSN: str = ""
    
    response_model_chain: list[str] = [
    "openrouter/free"
]
    
    
   
    
settings=Settings()
