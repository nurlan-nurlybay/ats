"""Bilingual classical-NLP resume parser (Stage 2).

Public surface:
    from ats.ingestion.parser import parse_resume, ParsedResume
"""
from ats.ingestion.parser.pipeline import parse_resume
from ats.ingestion.parser.schema import (
    EducationEntry,
    ExperienceEntry,
    ParsedResume,
)

__all__ = ["parse_resume", "ParsedResume", "ExperienceEntry", "EducationEntry"]
