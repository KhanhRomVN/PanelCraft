import cv2
import numpy as np
from typing import List, Optional, Tuple
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
        except Exception as e:
            # Continue without OCR functionality
            self.ocr_model = None
    
    async def process_segments(self, image: np.ndarray, segments: List[SegmentData]) -> List[OCRResult]:
        """
        Perform OCR on segmented regions
        """
        if self.ocr_model is None:
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
                        confidence=1.0
                    ))
                    
                except Exception as e:
                    continue
            
            return ocr_results
            
        except Exception as e:
            return []
        
    async def verify_text_boxes(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray,
        min_confidence: float = 0.5
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Verify text boxes bằng OCR và filter boxes có text quality tốt
        
        Args:
            image: Ảnh gốc (RGB)
            boxes: Text boxes [x1, y1, x2, y2]
            min_confidence: Confidence threshold
        
        Returns:
            Tuple: (verified_boxes, texts, confidences)
        """
        if self.ocr_model is None:
            return boxes, [], []
        
        if len(boxes) == 0:
            return np.array([]), [], []
        
        try:
            verified_boxes = []
            verified_texts = []
            verified_confidences = []
            
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                cropped = image[y1:y2, x1:x2]
                
                if cropped.size == 0:
                    continue
                
                try:
                    text = await self._run_ocr(cropped)
                    
                    # Check if text is valid
                    if text and len(text.strip()) > 0 and text != "[OCR ERROR]":
                        # Simple quality check
                        is_valid = self._is_valid_text(text)
                        
                        if is_valid:
                            verified_boxes.append(box)
                            verified_texts.append(text)
                            verified_confidences.append(1.0)
                            
                except Exception as e:
                    continue
            
            verified_boxes = np.array(verified_boxes) if verified_boxes else np.array([])
            
            return verified_boxes, verified_texts, verified_confidences
            
        except Exception as e:
            return boxes, [], []
    
    def _is_valid_text(self, text: str) -> bool:
        """
        Check if text is valid (not noise)
        """
        text = text.strip()
        
        if len(text) == 0:
            return False
        
        # Check nếu text quá ngắn và chỉ có ký tự đặc biệt
        if len(text) < 2:
            special_chars = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`"
            if all(c in special_chars for c in text):
                return False
        
        # Check nếu có ít nhất 1 ký tự alphabet hoặc CJK
        has_meaningful_char = any(
            c.isalpha() or 
            '\u4e00' <= c <= '\u9fff' or  # Chinese
            '\u3040' <= c <= '\u309f' or  # Hiragana
            '\u30a0' <= c <= '\u30ff' or  # Katakana
            '\uac00' <= c <= '\ud7af'     # Korean
            for c in text
        )
        
        return has_meaningful_char
    
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
            return "[OCR ERROR]"