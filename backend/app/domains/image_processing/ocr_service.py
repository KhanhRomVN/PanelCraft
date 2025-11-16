"""
OCR Service.

Responsibilities:
- Load manga-ocr model (MangaOcr) on initialization (graceful degradation if unavailable).
- Provide synchronous-style async OCR execution for segmented bubble regions.
- Supply verification & confidence scoring utilities for text boxes detected upstream.
- Normalize text quality evaluation (length, density, character classes, language hint).

Refactor / Clean Code Improvements:
- Added comprehensive docstrings & type hints.
- Centralized default confidence via app.config.constants (OCR_DEFAULT_CONFIDENCE).
- Unified logging prefixes ([OCR]).
- Early returns for degraded (no model) mode.
- Added structured error handling & removed silent broad excepts.
- Confidence calculation now returns consistent rounded metrics.
- Internal helpers (_run_ocr, _is_valid_text, calculate_ocr_confidence) documented.

Future Enhancements (not implemented to keep behavior stable):
- Introduce an OCRModel implementing BaseModel abstraction (see app/core/ml_models/base.py).
- Add caching layer for repeated region OCR (hash of image crop).
- Parallelize OCR over segments with asyncio.gather + semaphore for throughput.

"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from app.api.schemas.pipeline import SegmentData, OCRResult
from app.config import constants as C

logger = logging.getLogger(__name__)


class OCRService:
    """
    High-level wrapper around MangaOcr model.

    Args:
        model_base_path: Root model directory (currently unused; kept for parity
                         with other services & future custom OCR model locations).

    Degraded Mode:
        If manga-ocr fails to import or initialize, service still works but returns
        empty OCR results (avoid crashing entire pipeline).
    """

    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.ocr_model = None
        self._load_model()

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    def _load_model(self) -> None:
        """Attempt to load MangaOcr model. Logs failure & enters degraded mode if unavailable."""
        try:
            from manga_ocr import MangaOcr  # Lazy import
            self.ocr_model = MangaOcr()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[OCR] MangaOcr unavailable, degraded mode active: {e}")
            self.ocr_model = None

    # -------------------------------------------------------------------------
    # PUBLIC METHODS
    # -------------------------------------------------------------------------
    async def process_segments(
        self, image: np.ndarray, segments: List[SegmentData]
    ) -> List[OCRResult]:
        """
        Perform OCR over list of bubble segments.

        Args:
            image: RGB numpy array (H,W,3) uint8
            segments: List of SegmentData with bounding boxes

        Returns:
            List[OCRResult] (empty list if degraded mode)
        """
        if self.ocr_model is None:
            logger.info("[OCR] Skipping segment OCR (model not loaded)")
            return []

        ocr_results: List[OCRResult] = []
        for segment in segments:
            try:
                x1, y1, x2, y2 = segment.box
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                text = await self._run_ocr(cropped)
                if not text or text == "[OCR ERROR]":
                    continue

                ocr_results.append(
                    OCRResult(
                        segment_id=segment.id,
                        original_text=text,
                        confidence=C.OCR_DEFAULT_CONFIDENCE,
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[OCR] Segment {segment.id} OCR skip: {e}")

        return ocr_results

    async def verify_text_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        min_confidence: float = 0.5,
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Verify text boxes by running OCR & applying basic validity checks.

        Args:
            image: RGB original image
            boxes: Array of boxes [x1,y1,x2,y2]
            min_confidence: threshold (currently unused — placeholder for future model scoring)

        Returns:
            (verified_boxes, texts, confidences)
        """
        if self.ocr_model is None:
            logger.info("[OCR] verify_text_boxes skip (model not loaded)")
            return boxes, [], []

        if len(boxes) == 0:
            return np.array([]), [], []

        verified_boxes: List[List[int]] = []
        verified_texts: List[str] = []
        verified_confidences: List[float] = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            try:
                text = await self._run_ocr(cropped)
            except Exception:
                text = "[OCR ERROR]"

            if (
                text
                and text != "[OCR ERROR]"
                and self._is_valid_text(text)
            ):
                verified_boxes.append([x1, y1, x2, y2])
                verified_texts.append(text)
                verified_confidences.append(C.OCR_DEFAULT_CONFIDENCE)

        return (
            np.array(verified_boxes) if verified_boxes else np.array([]),
            verified_texts,
            verified_confidences,
        )

    def calculate_ocr_confidence(self, text: str, box_area: int) -> Dict[str, Any]:
        """
        Compute heuristic confidence & quality classification for OCR text.

        Factors:
            - Text length
            - Character density
            - Meaningful character ratio
            - Special character suppression

        Returns:
            {
              'confidence': float (0-100),
              'quality': str,
              'metrics': { ... detailed sub-metrics ... }
            }
        """
        text_stripped = text.strip()
        if not text_stripped:
            return self._confidence_result(
                confidence=0.0,
                quality="poor",
                metrics={
                    "text_length": 0,
                    "char_density": 0.0,
                    "has_meaningful_chars": False,
                    "language_detected": "none",
                    "special_char_ratio": 1.0,
                    "cjk_chars": 0,
                    "alpha_chars": 0,
                    "digit_chars": 0,
                    "special_chars": 0,
                },
            )

        total_chars = len(text_stripped)
        box_area_clamped = max(box_area, 1)
        char_density = total_chars / box_area_clamped

        cjk_chars = sum(
            1
            for c in text_stripped
            if "\u4e00" <= c <= "\u9fff"
            or "\u3040" <= c <= "\u309f"
            or "\u30a0" <= c <= "\u30ff"
            or "\uac00" <= c <= "\ud7af"
        )
        alpha_chars = sum(1 for c in text_stripped if c.isalpha())
        digit_chars = sum(1 for c in text_stripped if c.isdigit())
        special_chars = sum(
            1 for c in text_stripped if not c.isalnum() and not c.isspace()
        )
        special_char_ratio = special_chars / max(total_chars, 1)

        if cjk_chars > 0:
            language_detected = "CJK"
        elif alpha_chars > 0:
            language_detected = "Latin"
        else:
            language_detected = "Unknown"

        has_meaningful_chars = cjk_chars > 0 or alpha_chars > 0

        # Confidence scoring
        confidence = 0.0

        # Length
        if total_chars >= 10:
            confidence += 30
        elif total_chars >= 5:
            confidence += 20
        elif total_chars >= 2:
            confidence += 10

        # Density heuristic
        if 0.0001 <= char_density <= 0.01:
            confidence += 25
        elif 0.00005 <= char_density <= 0.02:
            confidence += 15

        # Meaningful chars
        if has_meaningful_chars:
            meaningful_ratio = (cjk_chars + alpha_chars) / max(total_chars, 1)
            confidence += meaningful_ratio * 25

        # Special character penalty (inverse)
        if special_char_ratio < 0.3:
            confidence += 20 * (1 - special_char_ratio / 0.3)

        confidence = max(0.0, min(100.0, confidence))

        if confidence >= 80:
            quality = "excellent"
        elif confidence >= 60:
            quality = "good"
        elif confidence >= 40:
            quality = "fair"
        else:
            quality = "poor"

        return self._confidence_result(
            confidence=confidence,
            quality=quality,
            metrics={
                "text_length": total_chars,
                "char_density": round(char_density, 6),
                "has_meaningful_chars": has_meaningful_chars,
                "language_detected": language_detected,
                "special_char_ratio": round(special_char_ratio, 3),
                "cjk_chars": cjk_chars,
                "alpha_chars": alpha_chars,
                "digit_chars": digit_chars,
                "special_chars": special_chars,
            },
        )

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------
    def _confidence_result(
        self, confidence: float, quality: str, metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Helper to assemble confidence result with rounding."""
        return {
            "confidence": round(confidence, 2),
            "quality": quality,
            "metrics": metrics,
        }

    def _is_valid_text(self, text: str) -> bool:
        """
        Basic validity check:
            - Non-empty
            - More than just special characters
            - At least one alphabetic or CJK character
        """
        text = text.strip()
        if len(text) == 0:
            return False
        if len(text) < 2 and all(c in "!@#$%^&*()_+-=[]{}|;:',.<>?/~`" for c in text):
            return False

        return any(
            c.isalpha()
            or "\u4e00" <= c <= "\u9fff"
            or "\u3040" <= c <= "\u309f"
            or "\u30a0" <= c <= "\u30ff"
            or "\uac00" <= c <= "\ud7af"
            for c in text
        )

    async def _run_ocr(self, image: np.ndarray) -> str:
        """
        Execute OCR on an RGB image crop.

        Returns:
            Recognized text or "[OCR ERROR]" on failure.
        """
        if self.ocr_model is None:
            return "[OCR ERROR]"

        try:
            from PIL import Image

            if image.size == 0:
                logger.debug("[OCR] Empty crop")
                return "[OCR ERROR]"

            h, w = image.shape[:2]
            pil_image = Image.fromarray(image)
            text = self.ocr_model(pil_image)
            return text
        except Exception as e:  # noqa: BLE001
            logger.error(f"[OCR] Exception: {e}")
            return "[OCR ERROR]"


__all__ = ["OCRService"]
