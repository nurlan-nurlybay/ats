"""Bilingual section segmentation for resumes.

Header detection combines typography signals (font size > body median,
bold, ALL CAPS) with fuzzy text matching (rapidfuzz) against a bilingual
header dictionary. A line is treated as a header only if BOTH signals
agree — typography alone has too many false positives, and text alone
breaks on stylized layouts.
"""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from statistics import median

from rapidfuzz import fuzz

from ats.ingestion.parser.extract import Line


class Section(str, Enum):
    ABOUT = "about"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    SKILLS = "skills"
    PROJECTS = "projects"
    LANGUAGES = "languages"
    CONTACTS = "contacts"
    OTHER = "other"


HEADER_DICT: dict[Section, list[str]] = {
    Section.EXPERIENCE: [
        "опыт работы",
        "опыт",
        "experience",
        "work experience",
        "work history",
        "профессиональный опыт",
        "трудовой опыт",
        "профессиональная деятельность",
        "карьера",
        "employment",
        "employment history",
    ],
    Section.EDUCATION: [
        "образование",
        "education",
        "обучение",
        "academic background",
        "academic experience",
    ],
    Section.SKILLS: [
        "навыки",
        "skills",
        "ключевые навыки",
        "стек",
        "технологии",
        "technical skills",
        "hard skills",
        "skills & technologies",
        "technologies",
    ],
    Section.PROJECTS: [
        "проекты",
        "projects",
        "pet projects",
        "selected projects",
        "personal projects",
    ],
    Section.LANGUAGES: ["языки", "languages", "знание языков"],
    Section.ABOUT: ["о себе", "summary", "profile", "about", "обо мне"],
    Section.CONTACTS: ["контакты", "contacts", "контактная информация"],
}


def detect_header(line: Line, body_size: float) -> Section | None:
    """Return the Section this line is a header for, or None."""
    text = line.text.strip()
    if not text or len(text) > 40:
        return None

    typo_signal = (
        line.max_font_size >= body_size + 1.0
        or line.is_bold
        or (text.isupper() and len(text.split()) <= 5)
    )
    if not typo_signal:
        return None

    text_norm = text.lower().rstrip(":").strip()
    best_section: Section | None = None
    best_score = 0.0
    for section, aliases in HEADER_DICT.items():
        for alias in aliases:
            score = fuzz.ratio(text_norm, alias)
            if score > best_score:
                best_score = score
                best_section = section

    return best_section if best_score >= 85 else None


def segment(lines: list[Line]) -> dict[Section, list[Line]]:
    """State machine: split lines into sections."""
    if not lines:
        return {}

    body_size = median(line.max_font_size for line in lines)

    sections: dict[Section, list[Line]] = defaultdict(list)
    state = Section.ABOUT

    for line in lines:
        matched = detect_header(line, body_size)
        if matched is not None:
            state = matched
            continue
        sections[state].append(line)

    return dict(sections)
