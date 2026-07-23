from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config=SettingsConfigDict(env_file=".env", extra="ignore")
    
    APP_ENV: str
    
    HF_TOKEN: str
    MONGO_USER: str
    MONGO_PASSWORD:str
    MONGO_URI: str
    MONGO_DB:str
    OPENROUTER_API_KEY: str
    
    response_model_chain: list[str] = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
]
    
    
   
    
settings=Settings()
