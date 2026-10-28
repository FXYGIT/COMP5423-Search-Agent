"""项目唯一配置入口。环境变量覆盖 .env；各模块不再另设配置来源。"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseModel):
    """模型 API 配置。model 填服务商提供的作业指定模型 ID。"""

    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0)


EmbeddingCategory = Literal["encoder", "instruction"]
RetrievalMethod = Literal["sparse", "dense", "instruction", "hybrid"]


class EmbeddingConfig(BaseModel):
    """建库与查询共用的编码配置；两个模型的向量不能混存。模型名由 .env 指定。"""

    dense_model: str = ""
    instruction_model: str = ""
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True
    query_instruction: str = "Find evidence answering the query."

    def model_for(self, category: EmbeddingCategory) -> str:
        """根据编码类别选择模型，供建库、检索和评估共用。"""
        return self.dense_model if category == "encoder" else self.instruction_model


class ChromaConfig(BaseModel):
    """本地持久化目录和两类模型各自的 collection。"""

    path: Path = Path("chroma")
    dense_collection: str = "browsecomp_dense"
    instruction_collection: str = "browsecomp_instruction"

    def collection_for(self, category: EmbeddingCategory) -> str:
        return self.dense_collection if category == "encoder" else self.instruction_collection


class RetrievalConfig(BaseModel):
    """统一 retrieve 的默认方法、数量和稀疏索引位置；模型可按单次问题覆盖方法。"""

    method: RetrievalMethod = "hybrid"
    top_k: int = 5
    sparse_index_path: Path = Path("indexes/bm25")
    rrf_constant: int = 60


class IndexingConfig(BaseModel):
    """离线建库选用的类别与分块参数；模型和集合从共享配置读取。"""

    category: EmbeddingCategory = "encoder"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    limit_documents: int | None = None


class DocumentConfig(BaseModel):
    """固定原文目录及默认单次阅读长度。"""

    corpus_path: Path = Path("data/browsecomp_plus_corpus")
    max_chars: int = 4000


class SkillsConfig(BaseModel):
    """技能根目录和脚本执行时限；算法和执行仍保留 TODO。"""

    root: Path = Path(__file__).parent / "skills" / "builtin"
    timeout_seconds: int = 30


class ContextConfig(BaseModel):
    """按保守估算控制请求大小；API 实际用量仅由响应 usage 统计。

    max_tokens 包含 system、工具定义和预留输出。摘要使用同一个模型，
    summary_max_tokens 限制辅助调用输出及保留摘要的大小。
    """

    max_tokens: int = Field(default=16000, gt=0)
    summary_trigger_tokens: int = Field(default=12000, gt=0)
    reserve_output_tokens: int = Field(default=2000, gt=0)
    evidence_max_tokens: int = Field(default=4000, gt=0)
    summary_max_tokens: int = Field(default=1500, gt=0)
    summary_enabled: bool = True

    @model_validator(mode="after")
    def validate_budget(self) -> "ContextConfig":
        if self.reserve_output_tokens >= self.max_tokens:
            raise ValueError("CONTEXT__RESERVE_OUTPUT_TOKENS 必须小于 MAX_TOKENS。")
        if self.summary_max_tokens >= self.max_tokens:
            raise ValueError("CONTEXT__SUMMARY_MAX_TOKENS 必须小于 MAX_TOKENS。")
        return self


class EvaluationConfig(BaseModel):
    """只保存评估独有配置；不重复 embedding、语料或 Chroma 配置。"""

    dataset_path: Path = Path("data/browsecomp_plus_val")
    split: str | None = None
    output_dir: Path = Path(".")
    predictions_path: Path = Path("test_prediction.jsonl")
    limit: int | None = None


class LoggingConfig(BaseModel):
    """统一文件日志设置；不配置控制台 handler。"""

    path: Path = Path("logs/search_agent.log")
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AppConfig(BaseSettings):
    """从 .env 或环境变量读取配置，例如 LLM__API_KEY。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
    )
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def get_config() -> AppConfig:
    """按需读取配置快照；入口读取后传给组件复用，导入模块不会读取 .env。

    不缓存，便于测试与不同入口读取各自环境。空可选环境变量使用默认值。
    离线评分不需要 LLM 密钥；只有创建真实聊天模型时才检查 LLM 配置。
    """
    return AppConfig()
