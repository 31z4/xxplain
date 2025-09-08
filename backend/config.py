from pydantic import HttpUrl, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.backend", extra="ignore")

    DEBUG: bool = False

    POSTGRES_DSN: PostgresDsn
    POSTGRES_DB_SCHEMA: str = "benchmarks/tpc-h/schema.sql"

    OPENAI_API_BASE_URL: HttpUrl = "https://api.openai.com/v1"
    OPENAI_API_KEY: SecretStr = ""
    OPENAI_API_MODEL: str = "gpt-5-mini"


settings = Settings()
