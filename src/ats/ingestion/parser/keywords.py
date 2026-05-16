"""Skill extraction via KeyBERT with the shared bge-m3 backbone.

Reuses the already-loaded SentenceTransformer to embed both the document
and candidate n-grams. The top-N most-similar n-grams (after MMR
diversification and a junk filter) become the candidate's `skills` list.
This is the "keyword extraction" deliverable from the assignment, done
without an LLM.
"""
from __future__ import annotations

import re
from functools import lru_cache

import spacy.lang.en.stop_words
import spacy.lang.ru.stop_words
from keybert import KeyBERT

from ats.core.logger import get_logger
from ats.ingestion.parser.embed import get_embedder

log = get_logger(__name__)

_COMMON_VERBS = {
    "developed",
    "designed",
    "implemented",
    "built",
    "created",
    "managed",
    "led",
    "worked",
    "supervised",
    "написал",
    "разработал",
    "разрабатывал",
    "создал",
    "внедрил",
    "проектировал",
    "руководил",
    "выполнял",
    "занимался",
}

_NUMERIC_RE = re.compile(r"^\d+$")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_DATE_LIKE_RE = re.compile(r"\d{2}[\.\-/]\d{2}")
_DIGIT_GROUPS_RE = re.compile(r"\d{2,}\s+\d{2,}")   # phone fragments like "708 669"
_CONTACT_FRAGMENTS = ("http", "https", "www", "gmail", "@", ".com", ".ru", ".kz", "telegram")


@lru_cache(maxsize=1)
def _stopwords() -> list[str]:
    return list(
        set(spacy.lang.ru.stop_words.STOP_WORDS)
        | set(spacy.lang.en.stop_words.STOP_WORDS)
    )


@lru_cache(maxsize=1)
def _keybert() -> KeyBERT:
    log.info("keybert_load")
    return KeyBERT(model=get_embedder())  # type: ignore[arg-type]


def _is_junk(phrase: str) -> bool:
    p = phrase.strip()
    if len(p) <= 2:
        return True
    if _NUMERIC_RE.match(p) or _YEAR_RE.match(p):
        return True
    if _DATE_LIKE_RE.search(p) or _DIGIT_GROUPS_RE.search(p):
        return True
    if p.lower() in _COMMON_VERBS:
        return True
    p_lower = p.lower()
    if any(frag in p_lower for frag in _CONTACT_FRAGMENTS):
        return True
    return False


def _dominated_by(phrase: str, name_tokens: set[str]) -> bool:
    """True if ≥50% of the phrase's tokens come from `name_tokens`."""
    if not name_tokens:
        return False
    parts = phrase.lower().split()
    if not parts:
        return False
    overlap = sum(1 for t in parts if t in name_tokens)
    return overlap / len(parts) >= 0.5


def extract_skills(
    text: str, top_n: int = 15, exclude_name: str | None = None
) -> list[str]:
    """Top-N skill phrases via KeyBERT. Pass the candidate name to drop
    phrases dominated by the candidate's own name/city tokens."""
    if not text.strip():
        return []
    kw_model = _keybert()
    raw: list[tuple[str, float]] = kw_model.extract_keywords(  # type: ignore[assignment]
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words=_stopwords(),
        use_mmr=True,
        diversity=0.5,
        top_n=top_n + 10,
    )
    name_tokens: set[str] = set()
    if exclude_name:
        name_tokens = {t.lower() for t in exclude_name.split() if len(t) > 1}

    out: list[str] = []
    seen: set[str] = set()
    for phrase, _score in raw:
        if _is_junk(phrase):
            continue
        if _dominated_by(phrase, name_tokens):
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= top_n:
            break
    return out
