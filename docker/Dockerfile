# Single multi-stage image, used by api / ui / worker / migrate services
# (they only differ in the `command:` they run).
#
# Layer strategy (cache-friendly):
#   1. apt-get build deps       — slow, almost never changes
#   2. pyproject.toml + stub src/ + pip install
#                               — heavy (~3 GB ML stack) but keyed only on
#                                 pyproject.toml, so src/ edits don't bust it
#   3. spaCy bilingual models   — moderate, stable
#   4. real src/                — fast, changes most often
#
# bge-m3 (~3.1 GB) is NOT baked in. It downloads on first container start
# into the `hf_cache` named volume mounted at /root/.cache/huggingface.

FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps: build-essential for any C extensions, libpq-dev for psycopg
# (we use asyncpg but keep libpq for SQLAlchemy compatibility), curl for
# the healthcheck.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── builder ──────────────────────────────────────────────────────────────
FROM base AS builder

# Layer A: deps install. Keyed on pyproject.toml + README.md only; survives
# all src/ edits. The setuptools build backend needs *something* importable
# at `src/ats/` to compute package metadata, so we create a stub
# __init__.py first. Real source comes in a later layer. Imports resolve
# via PYTHONPATH=/app/src (set in `base`), so we never need to re-run
# `pip install .` after the real src/ lands.
COPY pyproject.toml README.md ./
RUN mkdir -p src/ats && touch src/ats/__init__.py && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install .

# Layer B: spaCy bilingual models — needed by ats.ingestion.parser.ner.
# These are pip-installable wheel packages (~50 MB each) baked into the image.
RUN python -m spacy download ru_core_news_md && \
    python -m spacy download en_core_web_md

# Layer C: real source code. Cheap layer; src/ edits only invalidate from here.
COPY src/ ./src/

# ─── runtime ──────────────────────────────────────────────────────────────
FROM builder AS runtime

# Alembic + migrations are needed by the `migrate` service.
COPY alembic.ini ./
COPY migrations/ ./migrations/

# configs/ and data/ are bind-mounted by docker-compose so they can be
# hot-edited without rebuilding the image. (Documented in plan.md.)

EXPOSE 8000 8501

# Default to the API; the `ui` service overrides with `streamlit run …`.
CMD ["uvicorn", "ats.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
