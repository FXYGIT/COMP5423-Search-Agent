"""最小运行配置：只读取模型连接信息，不管理日志、预算或业务会话。"""

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseModel):
    """模型 API 配置。model 填服务商提供的作业指定模型 ID。"""

    api_key: SecretStr
    base_url: str
    model: str


class AppConfig(BaseSettings):
    """从 .env 或环境变量读取配置，例如 LLM__API_KEY。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    llm: LlmConfig


def get_config() -> AppConfig:
    """在程序启动时读取一次配置；导入 core 不会自动读取 .env。"""
    return AppConfig()
