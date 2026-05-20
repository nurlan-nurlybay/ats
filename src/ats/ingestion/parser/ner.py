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
#
# Precision over recall. Rules (Stage 7):
#   * 2 OR 3 tokens (no 1, no 4+).
#   * Each token ≥ 2 characters.
#   * All tokens uniformly Capitalized (Foo) OR uniformly ALLCAPS (FOO).
#     Mixed styles within one name fail validation.
#   * Cyrillic AND Latin scripts; a single internal hyphen permitted
#     (Анна-Мария / Mary-Jane).
#   * No digits, punctuation, or other noise tokens.
#   * If no candidate passes, return None — the UI renders "Not Found".

_NAME_TOKEN_CAPITAL = re.compile(
    r"^[A-ZА-ЯЁ][a-zа-яё]+(?:[-‑][A-ZА-ЯЁ][a-zа-яё]+)?$",
    re.UNICODE,
)
_NAME_TOKEN_ALLCAPS = re.compile(
    r"^[A-ZА-ЯЁ]{2,}(?:[-‑][A-ZА-ЯЁ]{2,})?$",
    re.UNICODE,
)


def _validate_name(candidate: str) -> str | None:
    """Return `candidate` only if it matches the strict Stage-7 name rules."""
    if not candidate:
        return None
    tokens = candidate.split()
    if not 2 <= len(tokens) <= 3:
        return None
    if any(len(t) < 2 for t in tokens):
        return None
    if all(_NAME_TOKEN_CAPITAL.fullmatch(t) for t in tokens):
        return candidate
    if all(_NAME_TOKEN_ALLCAPS.fullmatch(t) for t in tokens):
        return candidate
    return None


def find_name(top_lines: list[Line]) -> str | None:  # noqa: F821 (forward ref)
    """Extract candidate name from CV header (first 5 lines).

    Two-step matching, both gated by `_validate_name`:
      1. Run both spaCy models on the header; check each PER/PERSON entity
         (just its first newline-segment — entities can spill into the next
         line of city/role).
      2. Try the first 3 lines verbatim.

    Returns None if no candidate clears the strict rules.
    """
    if not top_lines:
        return None

    snippet = "\n".join(line.text for line in top_lines[:5])
    nlp_ru, nlp_en = _load_models()

    for nlp in (nlp_ru, nlp_en):
        doc = nlp(snippet)
        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON"):
                clean = ent.text.split("\n")[0].strip()
                validated = _validate_name(clean)
                if validated is not None:
                    return validated

    for line in top_lines[:3]:
        first = line.text.split("\n")[0].strip()
        validated = _validate_name(first)
        if validated is not None:
            return validated

    return None


# --- Date extraction ---
#
# Three patterns, applied in priority order. Spans matched by earlier
# patterns are excluded from later ones to avoid double-counting (a
# `Jun 2025 - Dec 2024` shouldn't ALSO produce two single-date hits).

# English + Russian month names. Russian uses nominative + common inflected
# forms (Январь, Январе, Января, etc.) since CVs vary.
_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r"|январ[ьяе]|феврал[ьяе]|март[а]?|апрел[ьяе]|ма[яй]|июн[ьяе]|июл[ьяе]"
    r"|август[а]?|сентябр[ьяе]|октябр[ьяе]|ноябр[ьяе]|декабр[ьяе]"
)

# "Present" markers — English + Russian variants.
_PRESENT = (
    r"present|now|current(?:\s+time)?|currently|"
    r"настоящее\s+время|по\s+наст\.?|по\s+настоящее"
)

# Date atom: any of `YYYY`, `MM/YYYY`, `MM-YYYY`, `Month YYYY`.
_DATE_ATOM = (
    r"(?:(?:0?[1-9]|1[0-2])[/.\-](?:19|20)\d{2}"
    r"|(?:" + _MONTHS + r")\s+(?:19|20)\d{2}"
    r"|(?:19|20)\d{2})"
)

# 1. Plain numeric-year range — kept for backward compat.
_DATE_RANGE_RE = re.compile(
    r"\b(?P<start>(?:19|20)\d{2})\s*[—–\-]\s*"
    r"(?P<end>(?:19|20)\d{2}|" + _PRESENT + r")",
    re.IGNORECASE,
)

# 2. Month-year or numeric-month-year range, both sides, with either side
#    possibly being a "present" marker.
_MONTH_YEAR_RANGE_RE = re.compile(
    r"(?P<start>" + _DATE_ATOM + r")"
    r"\s*[—–\-→]\s*"
    r"(?P<end>" + _DATE_ATOM + r"|" + _PRESENT + r")",
    re.IGNORECASE,
) 

# 3. Standalone date — graduation year / single mention. Catches single
#    `YYYY`, `Month YYYY`, or `MM/YYYY` that aren't part of a matched range.
_SINGLE_DATE_RE = re.compile(_DATE_ATOM, re.IGNORECASE)


def _normalize_end(raw: str) -> str:
    """Collapse present markers to the literal `present`; leave dates as-is."""
    low = raw.lower().strip()
    if re.match(_PRESENT, low, re.IGNORECASE):
        return "present"
    return raw.strip()


def extract_date_ranges(text: str) -> list[tuple[str | None, str | None]]:
    """Return list of (start, end) date-string tuples in order of appearance.

    `start` may be None for single dates (e.g., a graduation year), in which
    case the date sits in `end`. `end` is either a date string or the
    literal `"present"`.

    Two-sided ranges are matched first (numeric, then month-year); their
    spans are recorded so single-date scanning doesn't double-count tokens
    already consumed by a range.
    """
    ranges: list[tuple[str | None, str | None]] = []
    consumed: list[tuple[int, int]] = []

    # The broader month-year regex subsumes _DATE_RANGE_RE (bare YYYY is a
    # valid _DATE_ATOM). Iterating both would double-count.
    for m in _MONTH_YEAR_RANGE_RE.finditer(text):
        ranges.append((m.group("start").strip(), _normalize_end(m.group("end"))))
        consumed.append(m.span())

    for m in _SINGLE_DATE_RE.finditer(text):
        # Skip dates already consumed by a range.
        if any(s <= m.start() < e for s, e in consumed):
            continue
        ranges.append((None, m.group(0).strip()))

    return ranges


# Re-export type for type hints in this module
from ats.ingestion.parser.extract import Line  # noqa: E402
