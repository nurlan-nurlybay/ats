"""Deterministic Skills-section parser.

Real CVs structure skills as:

    • Programming Languages: Python, SQL, Kotlin
    • AI/ML skills: SciKitLearn, PyTorch, OpenAI SDK, NLTK
    Языки и фреймворки: Python (FastAPI, Django), SQL, Bash

This module walks the lines under a detected `Section.SKILLS` header and
emits a clean, deduplicated list of skill phrases. Handles:

  - `category: item1, item2, item3`        — category prefix discarded
  - `• item1` / `- item1` / `* item1`     — bullet prefix
  - `item1, item2, item3`                  — plain comma list
  - mixed: category header followed by bullet items on subsequent lines
  - line-wraps within a category (continuation lines with no `:` are
    appended to the previous category's item list)

When the SKILLS section is empty or contains <3 parsed items, the
pipeline falls back to KeyBERT (`extract_skills` in keywords.py).
"""
from __future__ import annotations

import re

from ats.ingestion.parser.extract import Line

# Category header: `Programming Languages: …` or `AI/ML skills: …`.
# Includes Cyrillic + Latin word chars, ampersand, slash, plus, dash, space.
# Limit to 50 chars on the category name to avoid swallowing entire sentences.
_CATEGORY_RE = re.compile(r"^\s*([\w\s&/+\-\.]{2,50})\s*[:：]\s*(.+)$")

# Leading bullet character.
_BULLET_RE = re.compile(r"^\s*[•∙·●▪▶▸‣–—\-*+]\s*")

# Skill item splitter — comma, slash, semicolon, pipe, or 2+ spaces.
_SPLITTERS = re.compile(r"\s*[,;|]\s*|\s{2,}")

# Reject tokens that look like noise (years, urls, phone fragments, etc.).
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_DIGIT_HEAVY_RE = re.compile(r"^\d{2,}[\s\-./]*\d*$")
_CONTACT_TOKENS = ("http", "https", "@", ".com", ".ru", ".kz", "gmail", "telegram")
_STOPWORD_PHRASES = {"and", "or", "и", "или", "etc", "etc.", "и т.д."}


def _is_noise(s: str) -> bool:
    s_low = s.lower().strip()
    if len(s) < 2 or len(s) > 60:
        return True
    if _YEAR_RE.match(s) or _DIGIT_HEAVY_RE.match(s):
        return True
    if s_low in _STOPWORD_PHRASES:
        return True
    if any(t in s_low for t in _CONTACT_TOKENS):
        return True
    return False


def _split_items(text: str) -> list[str]:
    """Split a `Python, SQL, Kotlin`-style list and clean each item.

    Preserves nested parens: `Python (FastAPI, Django)` should not split
    on the comma inside `(...)`. We do a depth-aware comma split rather
    than blindly applying the regex.
    """
    items: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in ",;|" and depth == 0:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf))
    # Secondary split on multi-space inside items (for typography-spaced lists).
    out: list[str] = []
    for it in items:
        out.extend(_SPLITTERS.split(it) if "  " in it else [it])
    return [x.strip().strip(".") for x in out if x.strip()]


def parse_skills_section(lines: list[Line]) -> list[str]:
    """Walk lines under a SKILLS header, return deduplicated skill phrases.

    Order is preserved (deduping keeps the first occurrence).
    """
    skills: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        # Strip a leading bullet to expose the category-or-list payload.
        text_nb = _BULLET_RE.sub("", text)
        m = _CATEGORY_RE.match(text_nb)
        payload = m.group(2) if m else text_nb
        for raw in _split_items(payload):
            key = raw.lower()
            if not _is_noise(raw) and key not in seen:
                seen.add(key)
                skills.append(raw)
    return skills
