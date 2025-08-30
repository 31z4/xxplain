from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.backend", extra="ignore")

    POSTGRES_DSN: PostgresDsn


settings = Settings()
