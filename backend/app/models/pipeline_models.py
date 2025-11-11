# app/models/pipeline_models.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

class ProcessingStep(str, Enum):
    SEGMENTATION = "segmentation"
    TEXT_DETECTION = "text_detection"
    OCR = "ocr"
    FULL_PIPELINE = "full_pipeline"

class SegmentData(BaseModel):
    id: int
    box: List[int]  # [x1, y1, x2, y2]
    score: float
    rectangle: Optional[List[int]] = None  # [x, y, w, h] - Largest inscribed rectangle

class OCRResult(BaseModel):
    segment_id: int
    original_text: str
    confidence: float = 0.0

class PipelineRequest(BaseModel):
    image_paths: List[str]
    model_base_path: str
    steps: List[ProcessingStep] = [ProcessingStep.FULL_PIPELINE]
    options: Optional[Dict[str, Any]] = None

class PipelineResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ImageResult(BaseModel):
    image_index: int
    original_path: str
    original_dimensions: Tuple[int, int]  # (width, height)
    cleaned_text_result: Optional[str] = None  # URL ảnh đã clean text
    segments: List[SegmentData] = []  # Segments với rectangles để Frontend vẽ outline
    rectangles: List[dict] = []  # THÊM: List rectangles metadata cho Frontend
    ocr_results: List[OCRResult] = []

class BatchResponse(BaseModel):
    request_id: str
    status: str  # processing, completed, error
    results: List[ImageResult] = []
    total_images: int
    processed_images: int