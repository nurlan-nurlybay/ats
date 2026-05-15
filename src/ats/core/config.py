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


class PathsConfig(BaseModel):
    data_raw: Path
    data_processed: Path
    fake_cvs: Path
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
    provider: Literal["openai", "huggingface"]
    model: str
    temperature: float
    max_tokens: int

    model_config = ConfigDict(frozen=True)


class ModelsConfig(BaseModel):
    semantic: SemanticModelConfig
    tfidf: TfidfConfig
    llm: LLMConfig

    model_config = ConfigDict(frozen=True)


class MatchingConfig(BaseModel):
    top_k: int
    default_strategy: Literal["semantic", "tfidf", "llm"]

    model_config = ConfigDict(frozen=True)


class VectorStoreConfig(BaseModel):
    index_type: str
    dimension: int

    model_config = ConfigDict(frozen=True)


class Settings(BaseModel):
    imap: IMAPSettings
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
        paths=PathsConfig(**raw["paths"]),
        models=ModelsConfig(**raw["models"]),
        matching=MatchingConfig(**raw["matching"]),
        vector_store=VectorStoreConfig(**raw["vector_store"]),
    )


settings: Settings = load_settings()
