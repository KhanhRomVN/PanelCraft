# app/models/pipeline_models.py
"""
DEPRECATED MODULE (Re-export only)

Original duplicated Pydantic models have been consolidated into:
    app/schemas/pipeline.py

This module now only re-exports the unified schema symbols for backward
compatibility. Prefer importing from `app.schemas.pipeline` in all new code.

Example (old code):
    from app.models.pipeline_models import PipelineRequest

Recommended (new code):
    from app.schemas.pipeline import PipelineRequest

Remove legacy imports gradually and then delete this file if no longer needed.
"""

from __future__ import annotations
from app.schemas.pipeline import (
    ProcessingStep,
    SegmentData,
    OCRResult,
    PipelineRequest,
    ImageResult,
    PipelineResponse,
    BatchResponse,
)

__all__ = [
    "ProcessingStep",
    "SegmentData",
    "OCRResult",
    "PipelineRequest",
    "ImageResult",
    "PipelineResponse",
    "BatchResponse",
]
