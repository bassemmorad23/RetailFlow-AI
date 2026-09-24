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
    SHOPIFY_CLIENT_ID: str = ""
    SHOPIFY_CLIENT_SECRET: str = ""
    SHOPIFY_REDIRECT_BASE_URL: str = "https://console-audacity-snort.ngrok-free.dev"
    INSTAGRAM_APP_ID: str = ""
    INSTAGRAM_APP_SECRET: str = ""
    INSTAGRAM_REDIRECT_BASE_URL: str = "https://console-audacity-snort.ngrok-free.dev"
    INSTAGRAM_WEBHOOK_VERIFY_TOKEN: str = "storeflow_verify_2026"
    META_APP_ID: str = ""
    META_APP_SECRET: str = ""
    META_REDIRECT_BASE_URL: str = "https://console-audacity-snort.ngrok-free.dev"
    META_WEBHOOK_VERIFY_TOKEN: str = "storeflow_verify_2026"
    META_FB_LOGIN_CONFIG_ID: str = ""
    WHATSAPP_ACCESS_TOKEN: str = ""
    WHATSAPP_PHONE_NUMBER_ID: str = ""
    WHATSAPP_BUSINESS_ACCOUNT_ID: str = ""
    WHATSAPP_APP_SECRET: str = ""
    SESSION_COOKIE_NAME: str = "sf_session"
    DASHBOARD_ORIGINS: str = "http://localhost:3000"  # comma-separated; add the real domain in production
    METRICS_TOKEN: str = ""  # required in production to read /metrics
    SESSION_COOKIE_SECURE: bool = False  # MUST be True in production (HTTPS)
    response_model_chain: list[str] = [
    "openrouter/free"
]
    
    
   
    
settings=Settings()
