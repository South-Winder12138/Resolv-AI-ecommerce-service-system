from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

MILVUS_COLLECTION_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,254}$"


def validate_mysql_database_url(value: SecretStr, setting_name: str) -> SecretStr:
    raw = value.get_secret_value().strip()
    message = f"{setting_name} requires a mysql+pymysql server URL with credentials, database and charset=utf8mb4"
    try:
        url = make_url(raw)
        valid = (
            url.drivername == "mysql+pymysql"
            and url.host
            and url.username
            and url.username.strip()
            and url.password
            and url.database
            and url.database.strip()
            and (url.port is None or 1 <= url.port <= 65535)
            and set(url.query) == {"charset"}
            and url.query.get("charset") == "utf8mb4"
        )
    except (ArgumentError, ValueError):
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)
    return SecretStr(raw)


def default_database_url() -> str:
    project_root = Path(__file__).resolve().parents[3]
    return f"sqlite:///{project_root / 'data' / 'resolv.db'}"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = Field(default_factory=default_database_url, min_length=1)


class Settings(DatabaseSettings):
    openai_api_key: SecretStr
    openai_model: str = "gpt-6-sol"
    openai_base_url: str = Field(min_length=1)


class KnowledgeDatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    knowledge_database_url: SecretStr

    @field_validator("knowledge_database_url")
    @classmethod
    def validate_mysql_url(cls, value: SecretStr) -> SecretStr:
        return validate_mysql_database_url(value, "knowledge_database_url")


class ConversationDatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", hide_input_in_errors=True
    )

    conversation_database_url: SecretStr

    @field_validator("conversation_database_url")
    @classmethod
    def validate_mysql_url(cls, value: SecretStr) -> SecretStr:
        return validate_mysql_database_url(value, "conversation_database_url")


class ConversationExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="CONVERSATION_"
    )
    owner_id: str = Field(default="local", min_length=1)
    lease_seconds: float = Field(default=120, gt=0)
    heartbeat_seconds: float = Field(default=30, gt=0)
    state_schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def safe_lease(self):
        if self.lease_seconds <= self.heartbeat_seconds:
            raise ValueError("lease must exceed heartbeat interval")
        return self


class ContextSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="CONTEXT_"
    )
    model_context_window_tokens: int = Field(gt=0)
    tokenizer_profile: Literal["cl100k_base", "o200k_base"]
    app_max_input_tokens: int = Field(default=12000, gt=0)
    provider_input_overhead_tokens: int = Field(default=0, ge=0)
    history_soft_budget_tokens: int = Field(default=6000, gt=0)
    summary_trigger_ratio: float = Field(default=0.8, gt=0, le=1)
    recent_history_target_tokens: int = Field(default=2000, ge=0)
    summary_max_tokens: int = Field(default=800, gt=0)
    knowledge_context_max_tokens: int = Field(default=3000, gt=0)
    reserved_output_tokens: int = Field(default=2000, gt=0)
    safety_margin_tokens: int = Field(default=1000, ge=0)

    @model_validator(mode="after")
    def positive_input_budget(self):
        if (
            self.model_context_window_tokens
            <= self.reserved_output_tokens + self.safety_margin_tokens
        ):
            raise ValueError(
                "model window must exceed output reserve and safety margin"
            )
        return self

    @property
    def input_limit(self):
        return min(
            self.app_max_input_tokens,
            self.model_context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )


class SummaryModelSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="SUMMARY_",
        hide_input_in_errors=True,
    )
    model_name: str = Field(min_length=1)
    api_key: SecretStr
    base_url: str = Field(min_length=1)
    model_context_window_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(default=2000, gt=0)
    input_max_tokens: int = Field(default=12000, gt=0)
    provider_input_overhead_tokens: int = Field(default=0, ge=0)
    safety_margin_tokens: int = Field(default=1000, ge=0)
    summary_max_tokens: int = Field(default=800, gt=0)
    timeout_seconds: float = Field(default=60, gt=0)

    @model_validator(mode="after")
    def positive_input_budget(self):
        if (
            self.model_context_window_tokens
            <= self.reserved_output_tokens + self.safety_margin_tokens
        ):
            raise ValueError(
                "summary window must exceed output reserve and safety margin"
            )
        return self

    @property
    def input_limit(self):
        return min(
            self.input_max_tokens,
            self.model_context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens,
        )


class KnowledgeChatSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="KNOWLEDGE_CHAT_"
    )

    enabled: bool = False
    model_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    collection: str = Field(
        default="resolv_knowledge_v2", pattern=MILVUS_COLLECTION_PATTERN
    )
    top_k: int = Field(default=5, ge=1, le=20)
    batch_size: int = Field(default=8, gt=0)
    passage_max_length: int = Field(default=512, gt=0, le=8192)
    query_max_length: int = Field(default=512, gt=0, le=8192)


class UnderstandingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="UNDERSTANDING_"
    )
    enabled: bool = False
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    timeout_seconds: float = Field(default=60, gt=0)
    state_schema_version: Literal[1] = 1
    provider_input_overhead_tokens: int | None = Field(default=6500, ge=0)


class BusinessExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_prefix="BUSINESS_"
    )
    max_agent_steps: int = Field(default=6, ge=1, le=6)
    provider_input_overhead_tokens: int = Field(default=6500, ge=0)


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = Field(
        default="resolv_knowledge_v1", pattern=MILVUS_COLLECTION_PATTERN
    )

    @field_validator("milvus_uri")
    @classmethod
    def validate_server_uri(cls, value: str) -> str:
        value = value.strip()
        uri = urlsplit(value)
        if uri.scheme not in {"http", "https"} or not uri.hostname or uri.port == 0:
            raise ValueError("milvus_uri must be an HTTP(S) Milvus server URI")
        return value


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="EMBEDDING_",
        populate_by_name=True,
    )

    model_name: str = Field(default="BAAI/bge-m3", validation_alias="BGE_M3_MODEL")
    device: str = "cpu"
    batch_size: int = Field(default=8, gt=0)
    use_fp16: bool = False
    query_max_length: int = Field(default=512, gt=0, le=8192)
    passage_max_length: int = Field(default=512, gt=0, le=8192)

    @field_validator("model_name", "device")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_name and device must be nonblank")
        return value
