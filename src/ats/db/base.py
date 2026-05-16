"""Async SQLAlchemy engine, session factory, and declarative base.

Every ORM model in `ats.db.models` subclasses `Base`. Every consumer of
the database (FastAPI handlers, utility scripts, the matching engine)
gets a session via `SessionLocal()` or, in FastAPI, `Depends(get_session)`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ats.core.config import settings


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


engine = create_async_engine(
    settings.db.url,
    echo=False,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Yields one session per request."""
    async with SessionLocal() as session:
        yield session
