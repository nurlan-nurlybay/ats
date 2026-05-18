"""Sync wrapper around the FastAPI surface.

Streamlit's runtime is sync, so we use `requests` rather than `httpx.AsyncClient`.
The 60-second timeout covers the LLM matcher's worst case (15 calls × ~4 s).
"""
from __future__ import annotations

import os
from typing import Any

import requests


class ApiError(RuntimeError):
    """Wraps the underlying transport / HTTP error for friendlier UI display."""


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.environ.get("ATS_API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    # ── public ─────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def list_vacancies(self) -> list[dict[str, Any]]:
        return self._get("/vacancies")

    def get_vacancy(self, vacancy_id: int) -> dict[str, Any]:
        return self._get(f"/vacancies/{vacancy_id}")

    def create_vacancy(
        self,
        *,
        title: str,
        description: str,
        experience: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "title": title,
            "description": description,
            "experience": experience or None,
        }
        return self._request("POST", "/vacancies", json=payload)

    def update_vacancy(
        self,
        vacancy_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        experience: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            k: v for k, v in {
                "title": title,
                "description": description,
                "experience": experience,
            }.items() if v is not None
        }
        return self._request("PUT", f"/vacancies/{vacancy_id}", json=payload)

    def delete_vacancy(self, vacancy_id: int) -> None:
        self._request("DELETE", f"/vacancies/{vacancy_id}")

    def list_candidates(self, *, include_eval: bool = False) -> list[dict[str, Any]]:
        return self._get("/candidates", params={"include_eval": str(include_eval).lower()})

    def get_candidate(self, candidate_id: int) -> dict[str, Any]:
        return self._get(f"/candidates/{candidate_id}")

    def upload_candidate(self, file_obj, filename: str) -> dict[str, Any]:
        """Stream a PDF/DOCX file to the API. `file_obj` is a streamlit
        UploadedFile or any object with `.read()`."""
        url = f"{self.base_url}/candidates"
        files = {"file": (filename, file_obj, "application/octet-stream")}
        try:
            resp = requests.post(url, files=files, timeout=self.timeout)
        except requests.RequestException as e:
            raise ApiError(f"upload failed: {e}") from e
        return self._unwrap(resp)

    def delete_candidate(self, candidate_id: int, *, keep_file: bool = False) -> None:
        self._request(
            "DELETE", f"/candidates/{candidate_id}",
            params={"keep_file": str(keep_file).lower()},
        )

    def get_recommendations(
        self,
        *,
        job_id: int | None = None,
        text: str | None = None,
        title: str | None = None,
        strategy: str,
        top_k: int,
        retriever: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"strategy": strategy, "top_k": top_k}
        if job_id is not None:
            params["job_id"] = job_id
        if text is not None:
            params["text"] = text
        if title:
            params["title"] = title
        if retriever:
            params["retriever"] = retriever
        return self._get("/recommendations", params=params)

    def trigger_pull(self) -> dict[str, Any]:
        """Trigger a Gmail pull + parse task. Returns ``{status, task_id}``."""
        return self._request("POST", "/ingestion/pull")

    def poll_pull_status(self, task_id: str) -> dict[str, Any]:
        """Check the status of a pull task."""
        return self._get(f"/ingestion/pull/{task_id}")

    # ── internals ──────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method, url, params=params, json=json, timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise ApiError(f"could not reach {url}: {e}") from e
        return self._unwrap(resp)

    def _unwrap(self, resp: requests.Response) -> Any:
        if not resp.ok:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except ValueError:
                pass
            raise ApiError(f"{resp.status_code} {resp.reason}: {detail}")
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()
