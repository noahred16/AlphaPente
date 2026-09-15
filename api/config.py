from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API settings, loaded from environment variables / the root .env file."""

    redis_url: str = "redis://localhost:6379/0"
    book_db_path: str = "book.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
