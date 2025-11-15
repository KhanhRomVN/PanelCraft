# app/schemas/pipeline.py
"""
Unified Pydantic schema definitions for the manga processing pipeline.

This module consolidates the duplicated models previously found in:
- app/api/models.py
- app/models/pipeline_models.py

Responsibilities:
- Define request/response data contracts (schemas) used by FastAPI endpoints.
- Provide clear documentation & typing for all data exchanged.
- Avoid business logic (keep pure data validation / serialization).

NOTE:
Remove or replace imports of app.api.models and app.models.pipeline_models with app.schemas.pipeline.
If backward compatibility is needed temporarily, those old modules can re-export these symbols.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class ProcessingStep(str, Enum):
    """
    Defines individual steps of the processing pipeline.
    FULL_PIPELINE means run all steps sequentially.
    """
    SEGMENTATION = "segmentation"
    TEXT_DETECTION = "text_detection"
    OCR = "ocr"
    FULL_PIPELINE = "full_pipeline"


class SegmentData(BaseModel):
    """
    Data for a detected bubble/text segment.

    box: [x1, y1, x2, y2] in absolute pixel coordinates.
    score: confidence score from segmentation model.
    rectangles: list of refined rectangles [x, y, w, h] possibly multiple per segment
                derived from text boxes or geometric expansion.
    """
    id: int
    box: List[int] = Field(..., min_items=4, max_items=4, description="Segment bounding box [x1,y1,x2,y2]")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    rectangles: List[List[int]] = Field(default_factory=list, description="List of rectangles [x,y,w,h]")


class OCRResult(BaseModel):
    """
    OCR output for one segment.
    original_text: raw OCR text (not cleaned).
    confidence: heuristic confidence score (0-1 or custom scale).
    """
    segment_id: int
    original_text: str
    confidence: float = 0.0


class PipelineRequest(BaseModel):
    """
    Request to process a batch of images through selected pipeline steps.

    image_paths: absolute or relative paths to images.
    model_base_path: root directory containing model subdirectories.
    steps: which pipeline steps to apply (default FULL_PIPELINE).
    options: optional key/value configuration overrides.
    """
    image_paths: List[str]
    model_base_path: str = Field(..., description="Base path containing model folders")
    steps: List[ProcessingStep] = Field(default_factory=lambda: [ProcessingStep.FULL_PIPELINE])
    options: Optional[Dict[str, Any]] = None


class ImageResult(BaseModel):
    """
    Result for one processed image.

    original_dimensions: (width, height).
    cleaned_text_result: URL (mounted static path) to cleaned image with text removed.
    segments: enriched segments (with rectangles).
    rectangles: flattened metadata for frontend consumption:
        {
          segment_id, rect_id, x, y, w, h
        }
    ocr_results: list of OCRResult objects for segments.
    text_outside_bubbles: list of dict metadata for text found outside bubble segments.
    """
    image_index: int
    original_path: str
    original_dimensions: Tuple[int, int]
    cleaned_text_result: Optional[str] = None
    segments: List[SegmentData] = Field(default_factory=list)
    rectangles: List[Dict[str, Any]] = Field(default_factory=list)
    ocr_results: List[OCRResult] = Field(default_factory=list)
    text_outside_bubbles: List[Dict[str, Any]] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    """
    Standard response for /process-images endpoint.
    data.results is a list[ImageResult] when success=True.
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """
    Status response for background batch processing.
    results may be partially filled while processing.
    """
    request_id: str
    status: str  # processing, completed, error
    results: List[ImageResult] = Field(default_factory=list)
    total_images: int
    processed_images: int


__all__ = [
    "ProcessingStep",
    "SegmentData",
    "OCRResult",
    "PipelineRequest",
    "ImageResult",
    "PipelineResponse",
    "BatchResponse",
]
