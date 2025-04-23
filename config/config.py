from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    DB_NAME: str = "postgres"
    DB_USER: str = "root"
    DB_PASSWORD: str = "New1234!"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432

    # Embedding settings
    EMBEDDING_DIM: int = 3072
    AZURE_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"
    AZURE_EMBEDDING_API_VERSION: str = "2024-02-15-preview"
    AZURE_EMBEDDING_ENDPOINT: str
    AZURE_EMBEDDING_API_KEY: str

    # Chat settings
    AZURE_CHAT_ENDPOINT: str = "gpt-4o"
    AZURE_CHAT_DEPLOYMENT: str
    AZURE_CHAT_API_VERSION: str
    AZURE_CHAT_API_KEY: str

    # Search settings
    DEFAULT_BM25_WEIGHT: float = 0.4
    DEFAULT_CONTENT_VECTOR_WEIGHT: float = 0.4
    DEFAULT_TITLE_VECTOR_WEIGHT: float = 0.2
    DEFAULT_TOP_K: int = 10
    DEFAULT_RERANK: bool = False
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    
    # Notion OAuth settings
    NOTION_CLIENT_ID: str = ""
    NOTION_CLIENT_SECRET: str = ""  # 환경 변수에서 로드하세요. 절대 하드코딩하지 마세요!
    NOTION_REDIRECT_URI: str = "http://localhost:3000/auth/notion/callback"
    JWT_SECRET_KEY: str = "your-super-secret-key-change-this-in-production"


    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
