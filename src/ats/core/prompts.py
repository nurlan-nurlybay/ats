"""Load and validate prompt templates from configs/prompts.yaml.

Single entry point: `load_prompts()`. The result is cached for the process
lifetime — prompts are immutable at runtime, so re-reading the YAML on every
LLM call would be wasted I/O.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROMPTS_PATH = _PROJECT_ROOT / "configs" / "prompts.yaml"


class LlmRerankPrompts(BaseModel):
    system_en: str = Field(min_length=20)
    system_ru: str = Field(min_length=20)
    user: str = Field(min_length=20)

    model_config = ConfigDict(frozen=True)


class Prompts(BaseModel):
    llm_rerank: LlmRerankPrompts

    model_config = ConfigDict(frozen=True)


@lru_cache(maxsize=1)
def load_prompts(path: Path | None = None) -> Prompts:
    p = path or _PROMPTS_PATH
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Prompts.model_validate(data)
