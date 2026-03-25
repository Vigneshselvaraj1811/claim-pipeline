from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""

    llm_provider: str = "groq"
    llm_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    pdf_max_pages: int = 100
    pdf_image_dpi: int = 150
    log_level: str = "INFO"
    max_upload_size_mb: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()