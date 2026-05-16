"""Resume parsing pipeline orchestrator.

Single public function: `parse_resume(path) -> ParsedResume`. Stitches
together the extract / segment / regex / NER / keywords / embed stages
into a typed Pydantic artifact ready to persist.
"""
from __future__ import annotations

import re
from pathlib import Path

from ats.core.logger import bind_context, clear_context, get_logger
from ats.ingestion.parser.embed import embed_text
from ats.ingestion.parser.extract import Line, extract_lines
from ats.ingestion.parser.keywords import extract_skills
from ats.ingestion.parser.ner import (
    Entity,
    extract_date_ranges,
    extract_entities,
    find_name,
)
from ats.ingestion.parser.schema import (
    EducationEntry,
    ExperienceEntry,
    ParsedResume,
)
from ats.ingestion.parser.segment import Section, segment

log = get_logger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{7,}\d)")
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _detect_languages(text: str) -> list[str]:
    cy = len(_CYRILLIC_RE.findall(text))
    la = len(_LATIN_RE.findall(text))
    total = cy + la
    if total == 0:
        return []
    langs: list[str] = []
    if cy / total > 0.05:
        langs.append("ru")
    if la / total > 0.05:
        langs.append("en")
    return langs


def _extract_phone(text: str) -> str | None:
    for m in _PHONE_RE.finditer(text):
        raw = m.group(1)
        digits = re.sub(r"\D", "", raw)
        if 7 <= len(digits) <= 15:
            return raw.strip()
    return None


def _build_experience_entries(
    section_lines: list[Line], entities: list[Entity]
) -> list[ExperienceEntry]:
    if not section_lines:
        return []
    text = "\n".join(line.text for line in section_lines)
    date_ranges = extract_date_ranges(text)
    orgs = [e for e in entities if e.label == "ORG"]

    if not date_ranges:
        if not text.strip():
            return []
        return [ExperienceEntry(raw=text[:1000])]

    entries: list[ExperienceEntry] = []
    for i, (start, end) in enumerate(date_ranges):
        org = orgs[i].text if i < len(orgs) else None
        entries.append(
            ExperienceEntry(
                organization=org, role=None, start=start, end=end, raw=text[:1000]
            )
        )
    return entries


def _build_education_entries(
    section_lines: list[Line], entities: list[Entity]
) -> list[EducationEntry]:
    if not section_lines:
        return []
    text = "\n".join(line.text for line in section_lines)
    date_ranges = extract_date_ranges(text)
    orgs = [e for e in entities if e.label == "ORG"]

    if not date_ranges:
        if not text.strip():
            return []
        inst = orgs[0].text if orgs else None
        return [EducationEntry(institution=inst, raw=text[:1000])]

    entries: list[EducationEntry] = []
    for i, (start, end) in enumerate(date_ranges):
        inst = orgs[i].text if i < len(orgs) else None
        entries.append(
            EducationEntry(institution=inst, start=start, end=end, raw=text[:1000])
        )
    return entries


def parse_resume(path: Path | str) -> ParsedResume:
    """End-to-end: file path → fully-populated ParsedResume.

    Raises:
        ValueError: unsupported extension or empty extraction.
    """
    path = Path(path).resolve()
    bind_context(file=path.name)
    try:
        log.info("parse_start")

        lines = extract_lines(path)
        if not lines:
            raise ValueError(f"No text extracted from {path}")
        cleaned_text = "\n".join(line.text for line in lines)
        log.info("extract_done", lines=len(lines), chars=len(cleaned_text))

        sections = segment(lines)
        log.info(
            "segment_done",
            sections={s.value: len(v) for s, v in sections.items()},
        )

        email_match = _EMAIL_RE.search(cleaned_text)
        email = email_match.group(0) if email_match else None
        phone = _extract_phone(cleaned_text)
        languages = _detect_languages(cleaned_text)

        name = find_name(lines[:10])

        exp_lines = sections.get(Section.EXPERIENCE, [])
        edu_lines = sections.get(Section.EDUCATION, [])
        exp_text = "\n".join(line.text for line in exp_lines)
        edu_text = "\n".join(line.text for line in edu_lines)
        exp_entities = extract_entities(exp_text) if exp_text else []
        edu_entities = extract_entities(edu_text) if edu_text else []

        experience = _build_experience_entries(exp_lines, exp_entities)
        education = _build_education_entries(edu_lines, edu_entities)
        log.info(
            "ner_done", experience=len(experience), education=len(education)
        )

        skills = extract_skills(cleaned_text, exclude_name=name)
        log.info("skills_done", count=len(skills))

        embedding = embed_text(cleaned_text)
        log.info("embed_done", dim=len(embedding))

        return ParsedResume(
            source_file=str(path),
            name=name,
            email=email,
            phone=phone,
            languages_detected=languages,
            skills=skills,
            experience=experience,
            education=education,
            raw_text=cleaned_text,
            embedding=embedding,
        )
    finally:
        clear_context()
