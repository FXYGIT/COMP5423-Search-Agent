from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseModel):
    """LLM 配置"""

    api_key: SecretStr = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)


class LoggingConfig(BaseModel):
    """日志配置"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: str = "./logs/core.log"


class AppConfig(BaseSettings):
    """应用配置，统一读取环境变量和 .env 文件。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LlmConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
