# Single multi-stage image, used by both the `api` and `ui` services
# (they only differ in the `command:` they run).
#
# Layer strategy:
#   1. apt-get build deps               — slow, rarely changes
#   2. pyproject.toml + source          — fast, changes most often
#   3. pip install + spaCy models       — moderate, changes when deps change
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

# Source must be present before `pip install .` because pyproject.toml
# requires src/ for the package metadata. README.md is required by the
# `[project].readme` declaration.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install .

# spaCy bilingual models — needed by ats.ingestion.parser.ner.
# These are pip-installable wheel packages (~50 MB each) baked into the image.
RUN python -m spacy download ru_core_news_md && \
    python -m spacy download en_core_web_md

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
