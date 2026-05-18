"""Vacancy list + detail + CRUD.

Read endpoints feed the Streamlit dropdown / preview pane. Write endpoints
(POST, PUT, DELETE) back the Streamlit Manage page so a recruiter can add
or edit job ads at runtime without touching JSON files on disk.

On create/update, the vacancy is embedded inline via the bge-m3 singleton
so subsequent semantic queries pick it up immediately.
"""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ats.api.dependencies import get_session
from ats.api.schemas import (
    VacancyCreate,
    VacancyDetail,
    VacancySummary,
    VacancyUpdate,
)
from ats.core.logger import get_logger
from ats.db.models import Vacancy
from ats.ingestion.parser.embed import embed_text

log = get_logger(__name__)
router = APIRouter(prefix="/vacancies", tags=["vacancies"])


def _embed_text(title: str, description: str) -> list[float]:
    return embed_text(f"{title}\n\n{description}")


def _to_detail(v: Vacancy) -> VacancyDetail:
    return VacancyDetail(
        id=v.id,
        title=v.title,
        description=v.description,
        experience=v.experience,
        source_filename=v.source_filename,
        created_at=v.created_at,
    )


# ─── READ ───────────────────────────────────────────────────────────────────

@router.get("", response_model=list[VacancySummary], summary="List all vacancies")
async def list_vacancies(
    session: AsyncSession = Depends(get_session),
) -> list[VacancySummary]:
    rows = (
        await session.execute(
            select(
                Vacancy.id,
                Vacancy.title,
                Vacancy.source_filename,
                Vacancy.experience,
            ).order_by(Vacancy.id)
        )
    ).all()
    return [
        VacancySummary(
            id=r.id,
            title=r.title,
            source_filename=r.source_filename,
            experience=r.experience,
        )
        for r in rows
    ]


@router.get(
    "/{vacancy_id}",
    response_model=VacancyDetail,
    responses={404: {"description": "Vacancy not found"}},
)
async def get_vacancy(
    vacancy_id: int,
    session: AsyncSession = Depends(get_session),
) -> VacancyDetail:
    v = await session.get(Vacancy, vacancy_id)
    if v is None:
        raise HTTPException(404, f"vacancy id={vacancy_id} not found")
    return _to_detail(v)


# ─── WRITE ──────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=VacancyDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new vacancy (auto-embeds)",
)
async def create_vacancy(
    payload: VacancyCreate,
    session: AsyncSession = Depends(get_session),
) -> VacancyDetail:
    source_filename = payload.source_filename or f"ui_{uuid4().hex[:8]}.json"
    # Reject duplicates explicitly — the unique constraint would otherwise
    # raise an IntegrityError caught only at flush time.
    existing = await session.execute(
        select(Vacancy.id).where(Vacancy.source_filename == source_filename)
    )
    if existing.scalar() is not None:
        raise HTTPException(409, f"source_filename {source_filename!r} already exists")

    embedding = _embed_text(payload.title, payload.description)
    v = Vacancy(
        title=payload.title,
        description=payload.description,
        experience=payload.experience,
        source_filename=source_filename,
        embedding=embedding,
    )
    session.add(v)
    await session.commit()
    await session.refresh(v)
    log.info("vacancy_created", id=v.id, title=v.title)
    return _to_detail(v)


@router.put(
    "/{vacancy_id}",
    response_model=VacancyDetail,
    summary="Update an existing vacancy (re-embeds if title/description changes)",
    responses={404: {"description": "Vacancy not found"}},
)
async def update_vacancy(
    vacancy_id: int,
    payload: VacancyUpdate,
    session: AsyncSession = Depends(get_session),
) -> VacancyDetail:
    v = await session.get(Vacancy, vacancy_id)
    if v is None:
        raise HTTPException(404, f"vacancy id={vacancy_id} not found")

    text_changed = False
    if payload.title is not None and payload.title != v.title:
        v.title = payload.title
        text_changed = True
    if payload.description is not None and payload.description != v.description:
        v.description = payload.description
        text_changed = True
    if payload.experience is not None:
        v.experience = payload.experience

    if text_changed:
        v.embedding = _embed_text(v.title, v.description)
        log.info("vacancy_reembedded", id=v.id)

    await session.commit()
    await session.refresh(v)
    log.info("vacancy_updated", id=v.id, text_changed=text_changed)
    return _to_detail(v)


@router.delete(
    "/{vacancy_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a vacancy",
    responses={404: {"description": "Vacancy not found"}},
)
async def delete_vacancy(
    vacancy_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    v = await session.get(Vacancy, vacancy_id)
    if v is None:
        raise HTTPException(404, f"vacancy id={vacancy_id} not found")
    await session.delete(v)
    await session.commit()
    log.info("vacancy_deleted", id=vacancy_id)
