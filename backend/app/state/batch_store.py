# app/state/batch_store.py
"""
In-memory batch processing store.

This centralizes the previous global variable `batch_processing` that was
declared inside the old monolithic endpoints module.

Responsibilities:
- Provide a single source of truth for batch/background job states.
- Easy future migration to Redis or another persistent backend.

Data Structure:
batch_processing_store = {
    request_id: {
        "status": "processing" | "completed" | "error",
        "total_images": int,
        "processed_images": int,
        "results": List[ImageResult],
        "error": Optional[str]
    }
}

Utility helpers are provided for clarity; core code currently accesses the
dict directly for minimal refactor disruption.
"""

from __future__ import annotations
from typing import Dict, Any

# Global in-memory store (replace with Redis in production)
batch_processing_store: Dict[str, Dict[str, Any]] = {}


def init_request(request_id: str, total_images: int) -> None:
    """Initialize a new batch request entry."""
    batch_processing_store[request_id] = {
        "status": "processing",
        "total_images": total_images,
        "processed_images": 0,
        "results": [],
    }


def update_progress(request_id: str, processed_images: int, results) -> None:
    """Update progress and optionally results for a batch request."""
    if request_id in batch_processing_store:
        batch_processing_store[request_id]["processed_images"] = processed_images
        batch_processing_store[request_id]["results"] = results


def mark_completed(request_id: str) -> None:
    """Mark a batch request as completed."""
    if request_id in batch_processing_store:
        batch_processing_store[request_id]["status"] = "completed"


def mark_error(request_id: str, error: str) -> None:
    """Mark a batch request as errored with an error message."""
    if request_id in batch_processing_store:
        batch_processing_store[request_id]["status"] = "error"
        batch_processing_store[request_id]["error"] = error
