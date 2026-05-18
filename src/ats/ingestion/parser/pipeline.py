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
from ats.ingestion.parser.skills_section import parse_skills_section

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


_BULLET_LEADING_RE = re.compile(r"^\s*[•∙·●▪▶▸‣\-*+]\s*")


def _strip_date_substrings(text: str) -> str:
    """Remove date-range and single-date matches from a line of text.

    Used to recover the role/title from a line like
        "Middle ML/AI Engineer Jun 2025 - Current Time"
    →   "Middle ML/AI Engineer"
    """
    from ats.ingestion.parser.ner import _MONTH_YEAR_RANGE_RE, _SINGLE_DATE_RE
    out = text
    for re_ in (_MONTH_YEAR_RANGE_RE, _SINGLE_DATE_RE):
        out = re_.sub("", out)
    # Collapse leftover whitespace/dashes that surrounded the dates.
    out = re.sub(r"\s*[—–\-→]\s*", " ", out)
    out = re.sub(r"\s+", " ", out).strip(" \t-—–:,.")
    return out


_GENERIC_ORG_TOKENS = {  # too-short or too-common to trust as a company name
    "ai", "api", "ml", "dl", "ui", "ux", "qa", "ux/ui", "ml/ai",
    "rest", "sql", "nlp", "css", "html", "js",
    "crud", "etl", "rag", "llm", "cv", "nlp",
}


def _looks_like_org_token(s: str) -> bool:
    """Heuristic: short single-token name-like residue (e.g., `Ryte.AI`, `VBox`)
    that's NOT a generic acronym — likely an org name on the anchor line.

    Stricter than just "short and capitalized" because Russian role titles
    (`Backend Инженер`) also fit that loose shape. We require ONE of:
      - a single token (no whitespace)
      - presence of `.` (typical of brand names like Ryte.AI, X.AI)
    """
    s = s.strip()
    if not s or len(s) > 20:
        return False
    if s.lower() in _GENERIC_ORG_TOKENS:
        return False
    if not s[0].isupper():
        return False
    tokens = s.split()
    if len(tokens) == 1:
        return True
    if len(tokens) <= 2 and "." in s:
        return True
    return False


def _find_org_in_line(text: str, orgs: list[str]) -> str | None:
    """Return the first ORG entity whose text appears (case-insensitive) in line.

    Filters out short generic acronyms (AI, API, ML, …) that spuriously
    register as ORG entities — they appear too often inside role / skill
    text to be useful as company names.
    """
    low = text.lower()
    for o in orgs:
        ol = o.strip().lower()
        if len(ol) <= 3 or ol in _GENERIC_ORG_TOKENS:
            continue
        # Token-boundary match (so "AI" doesn't match inside "ML/AI Engineer").
        if re.search(r"\b" + re.escape(ol) + r"\b", low):
            return o
    return None


def _build_experience_entries(
    section_lines: list[Line], entities: list[Entity]
) -> list[ExperienceEntry]:
    """Split the EXPERIENCE section into one entry per date-anchored block.

    Each block starts at a line containing a date range and extends to the
    next such line (or end of section). Bullets and continuation lines are
    appended to the block's `raw` field. The role is recovered by stripping
    the date substring from the anchor line. The org comes from any ORG NER
    span in the anchor line OR a "pending" ORG seen on a header line just
    before the anchor.
    """
    if not section_lines:
        return []
    orgs = [e.text for e in entities if e.label == "ORG"]

    entries: list[ExperienceEntry] = []
    current: dict | None = None
    # `pending_org` is the most-recently-seen company header on its own
    # line. It is STICKY: it persists across entries (a CV may list two
    # roles at the same company) and is only overwritten when a new
    # header line shows up.
    pending_org: str | None = None

    def _looks_like_generic_role_residue(s: str | None) -> bool:
        """`AI`, `API`, etc. — too short / generic to be a real role."""
        if not s:
            return True
        s_low = s.strip().lower()
        return len(s_low) <= 3 or s_low in _GENERIC_ORG_TOKENS

    for line in section_lines:
        text = line.text.strip()
        if not text:
            continue
        ranges = extract_date_ranges(text)
        if ranges:
            # Close out the previous entry, start a new one.
            if current is not None:
                entries.append(ExperienceEntry(**_finalize_exp(current)))
            start, end = ranges[0]
            role = _strip_date_substrings(text) or None
            line_org_hit = _find_org_in_line(text, orgs)
            # Heuristic: a date-stripped residue that's short and "name-like"
            # (≤ 2 tokens, ≤ 20 chars, looks like a proper noun) is more
            # likely an org-on-the-anchor-line than a role. This catches
            # `Ryte.AI Oct 2023 - Dec 2024`-style entries where the bilingual
            # NER fails to cleanly isolate the org.
            if role and _looks_like_org_token(role):
                org = role
                role = None
                pending_org = org  # carries forward if subsequent roles share it
            elif role and line_org_hit and role.lower() == line_org_hit.lower():
                org = line_org_hit
                role = None
                pending_org = line_org_hit
            else:
                # Default: header-line org wins; line-content NER is fallback.
                org = pending_org or line_org_hit
                if role and org and org.lower() in role.lower():
                    role = re.sub(re.escape(org), "", role, flags=re.IGNORECASE).strip(" \t-—–:,.") or None
            role_is_residue = _looks_like_generic_role_residue(role)
            current = {
                "organization": org,
                "role": None if role_is_residue else role,
                "start": start,
                "end": end,
                "lines": [text],
                "role_pending": role_is_residue,
            }
            # pending_org is sticky; subsequent entries at the same company
            # (e.g., VBox listing two roles) reuse it.
        else:
            stripped = _BULLET_LEADING_RE.sub("", text).strip()
            is_bullet = text != stripped
            if current is not None:
                current["lines"].append(text)
                # If we still need a role and this is a short non-bullet line,
                # claim it as the role (e.g., "Junior Data Analyst" right under
                # "Ryte.AI Oct 2023 - Dec 2024").
                if (
                    current.get("role_pending")
                    and not is_bullet
                    and 1 <= len(stripped.split()) <= 6
                ):
                    current["role"] = stripped
                    current["role_pending"] = False
            else:
                # Pre-entry header line — keep as candidate org.
                if not is_bullet and len(stripped.split()) <= 4:
                    pending_org = stripped

    if current is not None:
        entries.append(ExperienceEntry(**_finalize_exp(current)))
    return entries


def _finalize_exp(d: dict) -> dict:
    raw = "\n".join(d.pop("lines"))[:600]
    d.pop("role_pending", None)
    return {**d, "raw": raw}


def _build_education_entries(
    section_lines: list[Line], entities: list[Entity]
) -> list[EducationEntry]:
    """Same heuristic as experience, but: institution instead of org, and
    education blocks are usually a single line (`University, BSc, 2022`)."""
    if not section_lines:
        return []
    orgs = [e.text for e in entities if e.label == "ORG"]

    entries: list[EducationEntry] = []
    current: dict | None = None
    pending_inst: str | None = None

    for line in section_lines:
        text = line.text.strip()
        if not text:
            continue
        ranges = extract_date_ranges(text)
        if ranges:
            if current is not None:
                entries.append(EducationEntry(**_finalize_edu(current)))
            start, end = ranges[0]
            # Degree = the line minus dates minus the institution name.
            degree = _strip_date_substrings(text)
            inst = _find_org_in_line(text, orgs) or pending_inst
            if inst:
                degree = re.sub(re.escape(inst), "", degree, flags=re.IGNORECASE)
            degree = re.sub(r"\s+", " ", degree).strip(" \t-—–:,.")
            current = {
                "institution": inst,
                "degree": degree or None,
                "start": start,
                "end": end,
                "lines": [text],
            }
            pending_inst = None
        else:
            stripped = _BULLET_LEADING_RE.sub("", text).strip()
            is_bullet = text != stripped
            if current is None and not is_bullet and len(stripped.split()) <= 6:
                pending_inst = stripped
                continue
            if current is not None:
                current["lines"].append(text)
                if current.get("institution") is None:
                    found = _find_org_in_line(text, orgs)
                    if found is not None:
                        current["institution"] = found

    if current is not None:
        entries.append(EducationEntry(**_finalize_edu(current)))
    return entries


def _finalize_edu(d: dict) -> dict:
    raw = "\n".join(d.pop("lines"))[:600]
    return {**d, "raw": raw}


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

        # Prefer the deterministic Skills-section parser when the CV has
        # an explicit structured section. KeyBERT is a fallback for CVs
        # that don't (or whose section is too sparse to trust).
        skill_lines = sections.get(Section.SKILLS, [])
        section_skills = parse_skills_section(skill_lines) if skill_lines else []
        if len(section_skills) >= 3:
            skills = section_skills[:30]
            log.info("skills_done", count=len(skills), source="section")
        else:
            skills = extract_skills(cleaned_text, exclude_name=name)
            log.info("skills_done", count=len(skills), source="keybert_fallback")

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
