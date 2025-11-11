from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
    segmentation_result: Optional[str] = None  # URL to processed image
    text_detection_result: Optional[str] = None
    cleaned_text_result: Optional[str] = None  # Ảnh đã clean text từ text detection
    segments: List[SegmentData] = []
    ocr_results: List[OCRResult] = []

class BatchResponse(BaseModel):
    request_id: str
    status: str  # processing, completed, error
    results: List[ImageResult] = []
    total_images: int
    processed_images: int