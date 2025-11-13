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
    
    def calculate_ocr_confidence(self, text: str, box_area: int) -> dict:
        """
        Tính toán confidence score và quality metrics cho OCR text
        
        Returns:
            dict: {
                'confidence': float (0-100),
                'quality': str ('excellent', 'good', 'fair', 'poor'),
                'metrics': {
                    'text_length': int,
                    'char_density': float,
                    'has_meaningful_chars': bool,
                    'language_detected': str,
                    'special_char_ratio': float
                }
            }
        """
        text_stripped = text.strip()
        
        if not text_stripped:
            return {
                'confidence': 0.0,
                'quality': 'poor',
                'metrics': {
                    'text_length': 0,
                    'char_density': 0.0,
                    'has_meaningful_chars': False,
                    'language_detected': 'none',
                    'special_char_ratio': 1.0
                }
            }
        
        # Calculate metrics
        text_length = len(text_stripped)
        char_density = text_length / max(box_area, 1)
        
        # Count character types
        cjk_chars = sum(1 for c in text_stripped if 
                       '\u4e00' <= c <= '\u9fff' or  # Chinese
                       '\u3040' <= c <= '\u309f' or  # Hiragana
                       '\u30a0' <= c <= '\u30ff' or  # Katakana
                       '\uac00' <= c <= '\ud7af')    # Korean
        
        alpha_chars = sum(1 for c in text_stripped if c.isalpha())
        digit_chars = sum(1 for c in text_stripped if c.isdigit())
        special_chars = sum(1 for c in text_stripped if not c.isalnum() and not c.isspace())
        
        total_chars = len(text_stripped)
        special_char_ratio = special_chars / max(total_chars, 1)
        
        # Detect language (ưu tiên CJK vì manga thường là Japanese)
        if cjk_chars > 0:
            language_detected = 'CJK'
        elif alpha_chars > 0:
            language_detected = 'Latin'
        else:
            language_detected = 'Unknown'
        
        has_meaningful_chars = cjk_chars > 0 or alpha_chars > 0
        
        # Calculate confidence score (0-100)
        confidence = 0.0
        
        # Factor 1: Text length (max 30 points)
        if text_length >= 10:
            confidence += 30
        elif text_length >= 5:
            confidence += 20
        elif text_length >= 2:
            confidence += 10
        
        # Factor 2: Character density (max 25 points)
        if 0.0001 <= char_density <= 0.01:
            confidence += 25
        elif 0.00005 <= char_density <= 0.02:
            confidence += 15
        
        # Factor 3: Meaningful characters (max 25 points)
        if has_meaningful_chars:
            meaningful_ratio = (cjk_chars + alpha_chars) / max(total_chars, 1)
            confidence += meaningful_ratio * 25
        
        # Factor 4: Special character ratio (max 20 points)
        if special_char_ratio < 0.3:
            confidence += 20 * (1 - special_char_ratio / 0.3)
        
        # CRITICAL: Clamp confidence to 0-100 range
        confidence = max(0.0, min(100.0, confidence))
        
        # Determine quality level
        if confidence >= 80:
            quality = 'excellent'
        elif confidence >= 60:
            quality = 'good'
        elif confidence >= 40:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'confidence': round(confidence, 2),
            'quality': quality,
            'metrics': {
                'text_length': text_length,
                'char_density': round(char_density, 6),
                'has_meaningful_chars': has_meaningful_chars,
                'language_detected': language_detected,
                'special_char_ratio': round(special_char_ratio, 3),
                'cjk_chars': cjk_chars,
                'alpha_chars': alpha_chars,
                'digit_chars': digit_chars,
                'special_chars': special_chars
            }
        }
    
    async def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image region"""
        try:
            # Convert numpy array to PIL Image
            from PIL import Image
            
            # Check image quality
            if image.size == 0:
                logger.warning(f"[OCR] Empty image provided")
                return "[OCR ERROR]"
            
            # Log image info
            h, w = image.shape[:2]
            logger.debug(f"[OCR] Processing image: {w}x{h}px, size={image.size}bytes")
            
            pil_image = Image.fromarray(image)
            
            # Run OCR
            text = self.ocr_model(pil_image)
            
            logger.debug(f"[OCR] Result length: {len(text)} chars")
            
            return text
            
        except Exception as e:
            logger.error(f"[OCR] Exception: {str(e)}")
            return "[OCR ERROR]"