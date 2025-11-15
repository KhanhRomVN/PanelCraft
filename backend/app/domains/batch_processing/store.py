"""
Batch Processing Store (Authoritative Module)

Purpose:
    Provide a structured, extensible in-memory store for batch image processing requests.
    Supersedes legacy module: app/state/batch_store.py

Design Goals:
    - Clear typed representation of batch request state (dataclass).
    - Encapsulate mutation logic (init, progress update, completion, error).
    - Thread-safety via simple lock (future: replace with async primitives or Redis backend).
    - Graceful migration path to external persistence (Redis / DB) by swapping backend.

Public API:
    create_request(request_id: str, total_images: int) -> None
    update_progress(request_id: str, processed_images: int, results: list) -> None
    mark_completed(request_id: str) -> None
    mark_error(request_id: str, error: str) -> None
    get(request_id: str) -> BatchRequestState | None
    to_dict(request_id: str) -> dict | None
    all_requests() -> list[dict]

Migration Steps (from legacy):
    1. Replace imports:
           from app.state.batch_store import init_request
       ->  from app.domains.batch_processing.store import create_request
    2. Remove uses of raw dict access; use provided functions.
    3. Delete legacy file after confirming no imports remain.

Future Enhancements:
    - TTL eviction / cleanup routine.
    - Persistence layer adapter interface.
    - Event callbacks (on_progress, on_complete).
    - Async version / integration with task queue.

"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchRequestState:
    """Represents the state of a batch processing request."""
    request_id: str
    status: str = "processing"  # processing | completed | error
    total_images: int = 0
    processed_images: int = 0
    results: List[Any] = field(default_factory=list)
    error: Optional[str] = None

    def progress_pct(self) -> float:
        if self.total_images <= 0:
            return 0.0
        return round(self.processed_images / self.total_images * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["progress_pct"] = self.progress_pct()
        return data


class InMemoryBatchStore:
    """
    Thread-safe in-memory store (authoritative instance).
    Replace _store dict + locking strategy for external backend migration.
    """
    def __init__(self) -> None:
        self._store: Dict[str, BatchRequestState] = {}
        self._lock = threading.Lock()

    def create_request(self, request_id: str, total_images: int) -> None:
        with self._lock:
            if request_id in self._store:
                logger.warning("[BatchStore] Overwriting existing request_id=%s", request_id)
            self._store[request_id] = BatchRequestState(
                request_id=request_id,
                total_images=total_images,
            )
            logger.debug("[BatchStore] Created request %s (total=%d)", request_id, total_images)

    def update_progress(self, request_id: str, processed_images: int, results: List[Any]) -> None:
        with self._lock:
            state = self._store.get(request_id)
            if not state:
                logger.error("[BatchStore] update_progress missing request_id=%s", request_id)
                return
            state.processed_images = processed_images
            state.results = results
            logger.debug(
                "[BatchStore] Progress update %s: %d/%d (%.2f%%)",
                request_id,
                state.processed_images,
                state.total_images,
                state.progress_pct(),
            )

    def mark_completed(self, request_id: str) -> None:
        with self._lock:
            state = self._store.get(request_id)
            if not state:
                logger.error("[BatchStore] mark_completed missing request_id=%s", request_id)
                return
            state.status = "completed"
            logger.info("[BatchStore] Request %s completed", request_id)

    def mark_error(self, request_id: str, error: str) -> None:
        with self._lock:
            state = self._store.get(request_id)
            if not state:
                logger.error("[BatchStore] mark_error missing request_id=%s", request_id)
                return
            state.status = "error"
            state.error = error
            logger.warning("[BatchStore] Request %s errored: %s", request_id, error)

    def get(self, request_id: str) -> Optional[BatchRequestState]:
        with self._lock:
            return self._store.get(request_id)

    def to_dict(self, request_id: str) -> Optional[Dict[str, Any]]:
        state = self.get(request_id)
        return state.to_dict() if state else None

    def all_requests(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [state.to_dict() for state in self._store.values()]


# Global authoritative instance
batch_store = InMemoryBatchStore()


# Convenience functional API (mirrors legacy usage patterns)

def create_request(request_id: str, total_images: int) -> None:
    batch_store.create_request(request_id, total_images)


def update_progress(request_id: str, processed_images: int, results: List[Any]) -> None:
    batch_store.update_progress(request_id, processed_images, results)


def mark_completed(request_id: str) -> None:
    batch_store.mark_completed(request_id)


def mark_error(request_id: str, error: str) -> None:
    batch_store.mark_error(request_id, error)


def get_state(request_id: str) -> Optional[Dict[str, Any]]:
    return batch_store.to_dict(request_id)


def list_all() -> List[Dict[str, Any]]:
    return batch_store.all_requests()


__all__ = [
    "BatchRequestState",
    "create_request",
    "update_progress",
    "mark_completed",
    "mark_error",
    "get_state",
    "list_all",
    "batch_store",
]
