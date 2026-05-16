"""Layout-aware text extraction for PDF and DOCX resumes.

PDF path runs an x-axis histogram step over `extract_words` output to
detect a two-column gutter; if found, words are partitioned before line
reconstruction so the segmenter never sees text mashed across columns.
DOCX path walks the body in document order and handles tables by
flattening cells column-by-column.

Returns a list of `Line` objects carrying typography metadata (font
size, bold flag) that downstream stages use as section-header signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pdfplumber
from docx import Document


@dataclass(frozen=True)
class Line:
    text: str
    max_font_size: float
    is_bold: bool
    top: float
    column: int
    page: int


def extract_lines(path: Path) -> list[Line]:
    """Dispatch on file extension. Raises on unsupported types."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    raise ValueError(f"Unsupported file extension: {suffix} ({path})")


# --- PDF ---


def _extract_pdf(path: Path) -> list[Line]:
    all_lines: list[Line] = []
    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            words = page.extract_words(
                extra_attrs=["size", "fontname"],
                keep_blank_chars=False,
            )
            if not words:
                continue

            gutter_x = _detect_two_column_gutter(words, page.width)
            if gutter_x is None:
                lines = _words_to_lines(words, column=0, page=page_idx)
            else:
                left = [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter_x]
                right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter_x]
                lines = _words_to_lines(left, column=0, page=page_idx) + _words_to_lines(
                    right, column=1, page=page_idx
                )
            all_lines.extend(lines)
    return all_lines


def _detect_two_column_gutter(words: list[dict], page_width: float) -> float | None:
    """Return the x-coordinate of the gutter if a two-column layout is found, else None.

    Algorithm: histogram of word x-centers, smoothed; require exactly two
    peaks taller than 2× the median, separated by ≥20% of page width.
    The gutter is the trough between them.
    """
    if len(words) < 30:
        return None
    x_centers = np.array([(w["x0"] + w["x1"]) / 2 for w in words])
    bins = 50
    hist, edges = np.histogram(x_centers, bins=bins, range=(0, page_width))
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(hist, kernel, mode="same")

    med = float(np.median(smoothed))
    if med == 0:
        return None
    threshold = 2.0 * med

    peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if (
            smoothed[i] >= threshold
            and smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
        ):
            peaks.append(i)

    if len(peaks) < 2:
        return None

    p1, p2 = peaks[0], peaks[-1]
    if (edges[p2] - edges[p1]) / page_width < 0.20:
        return None

    trough_idx = p1 + int(np.argmin(smoothed[p1 : p2 + 1]))
    return float(edges[trough_idx])


def _words_to_lines(words: list[dict], column: int, page: int) -> list[Line]:
    """Cluster words by top-coordinate (±3pt) into lines."""
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    groups: list[list[dict]] = [[words[0]]]
    cur_top = words[0]["top"]
    TOL = 3.0
    for w in words[1:]:
        if abs(w["top"] - cur_top) <= TOL:
            groups[-1].append(w)
        else:
            groups.append([w])
            cur_top = w["top"]

    out: list[Line] = []
    for grp in groups:
        grp.sort(key=lambda x: x["x0"])
        text = " ".join(w["text"] for w in grp).strip()
        if not text:
            continue
        max_size = max(float(w.get("size", 10.0)) for w in grp)
        is_bold = any("bold" in str(w.get("fontname", "")).lower() for w in grp)
        out.append(
            Line(
                text=text,
                max_font_size=max_size,
                is_bold=is_bold,
                top=float(grp[0]["top"]),
                column=column,
                page=page,
            )
        )
    return out


# --- DOCX ---


def _extract_docx(path: Path) -> list[Line]:
    """Walk document body in order, handling paragraphs + tables."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    body = doc.element.body
    lines: list[Line] = []
    idx = 0

    def make_line(p, column: int) -> Line | None:
        nonlocal idx
        para = Paragraph(p, doc) if not isinstance(p, Paragraph) else p
        text = (para.text or "").strip()
        if not text:
            return None
        is_bold = any(bool(r.bold) for r in para.runs)
        sizes = [r.font.size.pt for r in para.runs if r.font.size is not None]
        max_size = max(sizes) if sizes else 10.0
        line = Line(
            text=text,
            max_font_size=max_size,
            is_bold=is_bold,
            top=float(idx),
            column=column,
            page=0,
        )
        idx += 1
        return line

    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            line = make_line(child, column=0)
            if line:
                lines.append(line)
        elif tag == "tbl":
            tbl = Table(child, doc)
            num_cols = len(tbl.columns) if tbl.columns else 0
            seen_cells: set[int] = set()
            for col_idx in range(num_cols):
                for row in tbl.rows:
                    if col_idx >= len(row.cells):
                        continue
                    cell = row.cells[col_idx]
                    cell_id = id(cell._tc)
                    if cell_id in seen_cells:
                        continue
                    seen_cells.add(cell_id)
                    for p in cell.paragraphs:
                        line = make_line(p, column=col_idx)
                        if line:
                            lines.append(line)
    return lines
