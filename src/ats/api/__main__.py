"""Entry point for `python -m ats.api`.

Used by docker-compose (default CMD in Dockerfile is `uvicorn ats.api.app:app`),
but also runnable directly during development.
"""
from __future__ import annotations

import uvicorn

from ats.api.app import app  # noqa: F401  (kept importable as ats.api.__main__:app)


def main() -> None:
    uvicorn.run(
        "ats.api.app:app",
        host="0.0.0.0",  # noqa: S104 — intentional; container exposes 8000 to host
        port=8000,
        workers=1,
        log_config=None,  # our structlog wins; uvicorn's default handler is silenced
    )


if __name__ == "__main__":
    main()
