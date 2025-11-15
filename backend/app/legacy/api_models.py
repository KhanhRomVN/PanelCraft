re# app/api/models.py
"""
DEPRECATED MODULE (Re-export Only)

Previous duplicated API schemas have been unified in:
    app/schemas/pipeline.py

This file now only re-exports those symbols for backward compatibility with
legacy imports such as:
    from app.api.models import PipelineRequest

Migrate all imports to:
    from app.schemas.pipeline import PipelineRequest

After all legacy imports are updated you can safely delete this file.
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
