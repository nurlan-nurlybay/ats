"""Hybrid bilingual NER.

Runs two spaCy pipelines (ru_core_news_md + en_core_web_md), merges
entities by span overlap, and augments with:
    1. PhraseMatcher over a small KZ/RU/global gazetteer
    2. Custom Matcher rules: TitleCase Latin pair + ALLCAPS Latin token

The augmentations are the defense against "code-switched" Latin entities
(e.g. "Forte Bank") embedded in Cyrillic context — the bilingual NER
nightmare the user flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher

from ats.core.logger import get_logger

log = get_logger(__name__)

GAZETTEER_DIR = Path(__file__).resolve().parents[4] / "data" / "gazetteers"


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int
    source: str


@lru_cache(maxsize=1)
def _load_models() -> tuple[Language, Language]:
    log.info("spacy_load", models=["ru_core_news_md", "en_core_web_md"])
    nlp_ru = spacy.load(
        "ru_core_news_md", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
    )
    nlp_en = spacy.load(
        "en_core_web_md", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
    )
    return nlp_ru, nlp_en


def _load_gazetteer(name: str) -> list[str]:
    path = GAZETTEER_DIR / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


@lru_cache(maxsize=2)
def _build_matchers(nlp_id: int) -> tuple[PhraseMatcher, Matcher]:
    nlp_ru, nlp_en = _load_models()
    nlp = nlp_ru if nlp_id == 0 else nlp_en

    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    orgs = _load_gazetteer("orgs.txt")
    unis = _load_gazetteer("unis.txt")
    if orgs:
        pm.add("ORG_GAZ", [nlp.make_doc(t) for t in orgs])
    if unis:
        pm.add("ORG_GAZ", [nlp.make_doc(t) for t in unis])

    m = Matcher(nlp.vocab)
    m.add(
        "LATIN_TITLE_PAIR",
        [
            [
                {"IS_TITLE": True, "IS_ASCII": True, "LENGTH": {">=": 2}},
                {"IS_TITLE": True, "IS_ASCII": True, "LENGTH": {">=": 2}},
            ]
        ],
    )
    m.add(
        "LATIN_ALLCAPS",
        [[{"IS_UPPER": True, "IS_ASCII": True, "LENGTH": {">=": 2, "<=": 6}}]],
    )
    return pm, m


def extract_entities(text: str) -> list[Entity]:
    """Run both spaCy models + rule augmentation; deduplicate by span overlap."""
    if not text.strip():
        return []
    nlp_ru, nlp_en = _load_models()

    out: list[Entity] = []
    for nlp_id, nlp in enumerate((nlp_ru, nlp_en)):
        lang = "ru" if nlp_id == 0 else "en"
        doc = nlp(text)
        for e in doc.ents:
            out.append(
                Entity(
                    text=e.text,
                    label=e.label_,
                    start=e.start_char,
                    end=e.end_char,
                    source=f"spacy_{lang}",
                )
            )
        pm, m = _build_matchers(nlp_id)
        for _match_id, s, end in pm(doc):
            span = doc[s:end]
            out.append(
                Entity(
                    text=span.text,
                    label="ORG",
                    start=span.start_char,
                    end=span.end_char,
                    source="rule_phrasematcher",
                )
            )
        for _match_id, s, end in m(doc):
            span = doc[s:end]
            out.append(
                Entity(
                    text=span.text,
                    label="ORG",
                    start=span.start_char,
                    end=span.end_char,
                    source="rule_matcher",
                )
            )

    return _dedupe(out)


def _dedupe(entities: list[Entity]) -> list[Entity]:
    """Keep longer spans; on overlap, prefer rule-based over spaCy."""
    entities = sorted(
        entities,
        key=lambda e: (
            e.start,
            -(e.end - e.start),
            0 if e.source.startswith("rule") else 1,
        ),
    )
    kept: list[Entity] = []
    for e in entities:
        skip = False
        for k in kept:
            ov_start = max(e.start, k.start)
            ov_end = min(e.end, k.end)
            if ov_end <= ov_start:
                continue
            ov_len = ov_end - ov_start
            shorter = min(e.end - e.start, k.end - k.start)
            if ov_len / shorter > 0.5:
                skip = True
                break
        if not skip:
            kept.append(e)
    return kept


# --- Name extraction ---

_NAME_REGEX = re.compile(
    r"^([A-ZА-ЯЁ][a-zа-яё]+(?:[-‑][A-ZА-ЯЁ][a-zа-яё]+)?"
    r"\s+[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)$",
    re.UNICODE,
)


def find_name(top_lines: list[Line]) -> str | None:  # noqa: F821 (forward ref)
    """Extract candidate name from CV header (first 5 lines).

    spaCy PER spans can spill into the line below (city, role) when the
    name and city sit on consecutive lines. We split entity text on
    newline and take just the first segment.
    """
    if not top_lines:
        return None

    snippet = "\n".join(line.text for line in top_lines[:5])
    nlp_ru, nlp_en = _load_models()

    best: str | None = None
    for nlp in (nlp_ru, nlp_en):
        doc = nlp(snippet)
        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON"):
                clean = ent.text.split("\n")[0].strip()
                words = clean.split()
                if 2 <= len(words) <= 4:
                    return clean
                if best is None and clean:
                    best = clean

    if best is not None:
        return best

    for line in top_lines[:3]:
        first = line.text.split("\n")[0].strip()
        match = _NAME_REGEX.match(first)
        if match:
            return match.group(1)
    return None


# --- Date extraction ---

_DATE_RANGE_RE = re.compile(
    r"\b(?P<start>(?:19|20)\d{2})\s*[—–\-]\s*"
    r"(?P<end>(?:19|20)\d{2}|present|now|настоящее время|по\s*наст\.?|по\s*настоящее)",
    re.IGNORECASE,
)


def extract_date_ranges(text: str) -> list[tuple[str, str]]:
    """Return list of (start_year, end_year_or_'present') tuples."""
    ranges: list[tuple[str, str]] = []
    for m in _DATE_RANGE_RE.finditer(text):
        start = m.group("start")
        end_raw = m.group("end").lower()
        end_match = re.match(r"(19|20)\d{2}", end_raw)
        end = end_match.group(0) if end_match else "present"
        ranges.append((start, end))
    return ranges


# Re-export type for type hints in this module
from ats.ingestion.parser.extract import Line  # noqa: E402
