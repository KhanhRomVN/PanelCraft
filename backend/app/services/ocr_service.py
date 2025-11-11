import cv2
import numpy as np
from typing import List, Optional
import logging

from app.models.pipeline_models import SegmentData, OCRResult

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.ocr_model = None
        self._load_model()
    
    def _load_model(self):
        """Load OCR model"""
        try:
            from manga_ocr import MangaOcr
            self.ocr_model = MangaOcr()
            logger.info("OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load OCR model: {e}")
            # Continue without OCR functionality
            self.ocr_model = None
    
    async def process_segments(self, image: np.ndarray, segments: List[SegmentData]) -> List[OCRResult]:
        """
        Perform OCR on segmented regions
        """
        if self.ocr_model is None:
            logger.warning("OCR model not available")
            return []
        
        try:
            ocr_results = []
            
            for segment in segments:
                try:
                    # Crop segment region
                    x1, y1, x2, y2 = segment.box
                    cropped = image[y1:y2, x1:x2]
                    
                    if cropped.size == 0:
                        continue
                    
                    # Run OCR
                    text = await self._run_ocr(cropped)
                    
                    ocr_results.append(OCRResult(
                        segment_id=segment.id,
                        original_text=text,
                        confidence=1.0  # manga-ocr doesn't provide confidence scores
                    ))
                    
                except Exception as e:
                    logger.error(f"OCR failed for segment {segment.id}: {e}")
                    continue
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return []
    
    async def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image region"""
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(image)
            
            # Run OCR
            text = self.ocr_model(pil_image)
            
            return text
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "[OCR ERROR]"