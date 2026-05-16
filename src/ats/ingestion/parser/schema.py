"""Pydantic schema for the parsed resume artifact."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExperienceEntry(BaseModel):
    organization: str | None = None
    role: str | None = None
    start: str | None = None
    end: str | None = None
    raw: str

    model_config = ConfigDict(frozen=True)


class EducationEntry(BaseModel):
    institution: str | None = None
    degree: str | None = None
    start: str | None = None
    end: str | None = None
    raw: str

    model_config = ConfigDict(frozen=True)


class ParsedResume(BaseModel):
    source_file: str
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    languages_detected: list[str] = []
    skills: list[str] = []
    experience: list[ExperienceEntry] = []
    education: list[EducationEntry] = []
    raw_text: str
    embedding: list[float]
