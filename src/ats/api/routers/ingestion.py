"""Ingestion control endpoint.

POST /ingestion/pull — triggers the pull-and-parse Celery task on demand.
Returns 202 Accepted with a task_id that can be polled for status.
GET  /ingestion/pull/{task_id} — check the status of a previously
     triggered pull task.
"""
from __future__ import annotations

from fastapi import APIRouter, status

from ats.core.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post(
    "/pull",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Gmail CV pull + parse (async via Celery)",
    response_model=dict,
)
async def trigger_pull() -> dict:
    from ats.ingestion.tasks import pull_and_parse_cvs

    result = pull_and_parse_cvs.delay()
    log.info("ingestion_pull_triggered", task_id=result.id)
    return {"status": "accepted", "task_id": result.id}


@router.get(
    "/pull/{task_id}",
    summary="Check status of a pull task",
    response_model=dict,
)
async def pull_status(task_id: str) -> dict:
    from celery.result import AsyncResult

    from ats.ingestion.tasks import app as celery_app

    result = AsyncResult(task_id, app=celery_app)
    response: dict = {"task_id": task_id, "status": result.state}
    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result)
    return response
