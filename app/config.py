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
    
    response_model_chain: list[str] = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    
    "poolside/laguna-s-2.1:free",
]
    
    
   
    
settings=Settings()
