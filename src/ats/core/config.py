"""Unified application settings.

Loads IMAP credentials from `.env` (via pydantic-settings) and ML / path
configuration from `configs/config.yaml`. Import the module-level
`settings` singleton from anywhere in the app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"


class IMAPSettings(BaseSettings):
    """IMAP credentials, loaded from `.env` at the repo root."""

    server: str
    port: int = 993
    user: str
    password: SecretStr

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="IMAP_",
        extra="ignore",
        frozen=True,
    )


class DBSettings(BaseSettings):
    """Postgres connection settings, loaded from `.env`.

    The URL is composed in Python so the password (a `SecretStr`) never
    has to live in a committed config file.
    """

    user: str
    password: SecretStr
    db: str
    host: str = "localhost"
    port: int = 5432

    @property
    def url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="POSTGRES_",
        extra="ignore",
        frozen=True,
    )


class LLMSecrets(BaseSettings):
    """LLM provider API keys, loaded from `.env`.

    Kept separate from `models.llm` (YAML, non-secret tunables) so secrets
    never leak into serialized config dumps.
    """

    dashscope_api_key: SecretStr

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        frozen=True,
    )


class PathsConfig(BaseModel):
    data_processed: Path
    cvs: Path
    vacancies: Path
    index_dir: Path

    model_config = ConfigDict(frozen=True)


class SemanticModelConfig(BaseModel):
    name: str
    embedding_dimension: int
    device: Literal["cuda", "cpu", "mps", "auto"]
    max_seq_length: int
    batch_size: int
    normalize_embeddings: bool

    model_config = ConfigDict(frozen=True)


class TfidfConfig(BaseModel):
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int

    model_config = ConfigDict(frozen=True)


class LLMConfig(BaseModel):
    provider: Literal["openai", "qwen", "huggingface"]
    model: str
    temperature: float
    max_tokens: int
    base_url: str | None = None

    model_config = ConfigDict(frozen=True)


class ModelsConfig(BaseModel):
    semantic: SemanticModelConfig
    tfidf: TfidfConfig
    llm: LLMConfig

    model_config = ConfigDict(frozen=True)


class LLMMatchingConfig(BaseModel):
    shortlist_size: int = 15
    max_concurrent: int = 5
    timeout_seconds: float = 30.0
    retry_attempts: int = 3

    model_config = ConfigDict(frozen=True)


class MatchingConfig(BaseModel):
    top_k: int
    default_strategy: Literal["semantic", "tfidf", "llm", "rrf"]
    llm: LLMMatchingConfig = LLMMatchingConfig()

    model_config = ConfigDict(frozen=True)


class VectorStoreConfig(BaseModel):
    index_type: str
    dimension: int

    model_config = ConfigDict(frozen=True)


class Settings(BaseModel):
    imap: IMAPSettings
    db: DBSettings
    llm_secrets: LLMSecrets
    paths: PathsConfig
    models: ModelsConfig
    matching: MatchingConfig
    vector_store: VectorStoreConfig

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


def load_settings(config_path: Path | None = None) -> Settings:
    """Build a Settings object from `.env` + `config.yaml`.

    Paths in the YAML are resolved relative to PROJECT_ROOT so callers do
    not need to care about the current working directory.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    for key, value in raw["paths"].items():
        p = Path(value)
        raw["paths"][key] = p if p.is_absolute() else (PROJECT_ROOT / p)

    return Settings(
        imap=IMAPSettings(),  # type: ignore
        db=DBSettings(),  # type: ignore
        llm_secrets=LLMSecrets(),  # type: ignore
        paths=PathsConfig(**raw["paths"]),
        models=ModelsConfig(**raw["models"]),
        matching=MatchingConfig(**raw["matching"]),
        vector_store=VectorStoreConfig(**raw["vector_store"]),
    )


settings: Settings = load_settings()
