"""
Image Processing Pipeline Orchestrator.

Responsibilities:
- Coordinate domain services: Segmentation, Text Detection, Inpainting, OCR.
- Execute a configurable ordered subset of processing steps on a single image.
- Produce a unified ImageResult containing intermediate visualization metadata.
- Maintain clean separation between orchestration logic and domain services.

Clean Code / Refactor Improvements:
- Added comprehensive module & class docstrings.
- Introduced explicit step handler methods (_run_segmentation, _run_text_detection, _run_ocr).
- Clear ProcessingContext dataclass to share intermediate artifacts safely.
- Centralized error handling & logging (consistent prefixes [Pipeline]).
- Removed inline duplicated logic; clarified variable lifetimes.
- Added type hints everywhere.
- Minimized deep nesting using early returns and helper methods.
- Prepared ground for future batch processing abstraction.

Future Enhancements (not implemented to keep behavior stable):
- Async concurrency for OCR & text detection (bounded semaphore).
- Pluggable pipeline step registry (dependency injection for custom steps).
- Caching of visualization artifacts.
- Streamed partial responses for long-running images.

NOTE:
This orchestrator keeps previous visualization side-effects (saving temp images) for backward compatibility.
"""

from __future__ import annotations

import os
import cv2
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from app.api.schemas.pipeline import (
    ProcessingStep,
    ImageResult,
    SegmentData,
    OCRResult,
)
from app.domains.image_processing.segmentation_service import SegmentationService
from app.domains.image_processing.text_detection_service import TextDetectionService
from app.domains.image_processing.ocr_service import OCRService
from app.domains.image_processing.inpainting_service import InpaintingService
from app.core.image_utils.processing import numpy_to_base64, save_temp_image  # noqa: F401 (numpy_to_base64 retained if used externally)
from app.config.settings import settings  # noqa: F401 (settings retained for future dynamic config)
# from app.config import constants as C  # (Optional future usage)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    """
    Shared intermediate artifacts across pipeline steps.

    Attributes:
        image_rgb: Loaded RGB image.
        segments: Segmentation results (without rectangles until text detection step).
        masks: Binary masks for each segment.
        rectangles_flat: Flattened rectangle metadata for frontend usage.
        cleaned_image: Image with text removed inside bubble segments (Step 2).
        visualizations: Dict of named visualization URLs saved to /temp.
        text_boxes_per_segment: List of arrays of per-segment detected text boxes.
        all_text_boxes: Global text boxes across entire image.
        all_text_scores: Confidence scores for global text boxes.
        text_boxes_outside: Global text boxes outside bubble segments.
    """
    image_rgb: Optional[np.ndarray] = None
    segments: List[SegmentData] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    rectangles_flat: List[Dict[str, Any]] = field(default_factory=list)
    cleaned_image: Optional[np.ndarray] = None
    visualizations: Dict[str, str] = field(default_factory=dict)
    text_boxes_per_segment: List[np.ndarray] = field(default_factory=list)
    all_text_boxes: np.ndarray = field(default_factory=lambda: np.array([]))
    all_text_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    text_boxes_outside: np.ndarray = field(default_factory=lambda: np.array([]))


class ImageProcessingPipelineService:
    """
    Orchestrates execution of manga page processing pipeline.

    Steps (see ProcessingStep Enum):
        SEGMENTATION  -> bubble segmentation + boundary visualization
        TEXT_DETECTION -> per-segment text boxes, rectangle refinement, cleaning, outside text processing
        OCR            -> OCR over bubble segments (after rectangles computed)
        FULL_PIPELINE  -> executes all above in fixed order

    Usage:
        service = ImageProcessingPipelineService(model_base_path)
        result = await service.process_single_image(path, steps=[...])
    """

    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.segmentation_service = SegmentationService(model_base_path)
        self.text_detection_service = TextDetectionService(model_base_path)
        self.ocr_service = OCRService(model_base_path)
        self.inpainting_service = InpaintingService(model_base_path)

    # -------------------------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # -------------------------------------------------------------------------
    async def process_single_image(
        self,
        image_path: str,
        steps: Optional[List[ProcessingStep]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> ImageResult:
        """
        Process a single image by executing the requested pipeline steps.

        Args:
            image_path: Path to image file.
            steps: List of ProcessingStep enums (defaults to FULL_PIPELINE).
            options: Future extension for dynamic overrides (currently unused).

        Returns:
            ImageResult with all collected artifacts.
        """
        steps = steps or [ProcessingStep.FULL_PIPELINE]
        options = options or {}

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ctx = ProcessingContext(image_rgb=image_rgb)

        # Base result scaffold
        result = ImageResult(
            image_index=0,
            original_path=image_path,
            original_dimensions=(image_rgb.shape[1], image_rgb.shape[0]),
            segments=[],
            ocr_results=[],
        )

        try:
            full_pipeline = ProcessingStep.FULL_PIPELINE in steps

            if full_pipeline or ProcessingStep.SEGMENTATION in steps:
                await self._run_segmentation(ctx, result)

            if full_pipeline or ProcessingStep.TEXT_DETECTION in steps:
                await self._run_text_detection(ctx, result)

            if full_pipeline or ProcessingStep.OCR in steps:
                await self._run_ocr(ctx, result)

            # Final assignments
            result.image_index = 0
            return result

        except Exception as e:  # noqa: BLE001
            logger.error("[Pipeline] Error processing image %s: %s", image_path, e)
            raise

    # -------------------------------------------------------------------------
    # STEP HANDLERS
    # -------------------------------------------------------------------------
    async def _run_segmentation(
        self, ctx: ProcessingContext, result: ImageResult
    ) -> None:
        """Execute segmentation step & store segments/masks + visualization."""
        logger.info("═══════════════════════════ [STEP1] - BUBBLES SEGMENTATION ═══════════════════════════")
        segments, _vis_placeholder, masks = await self.segmentation_service.process(
            ctx.image_rgb
        )
        ctx.segments = segments
        ctx.masks = masks
        result.segments = segments

        # Visualization (boundaries only)
        vis_boundaries = self.segmentation_service._create_step1_visualization(
            ctx.image_rgb.copy(), segments, masks
        )
        step1_vis_path = save_temp_image(vis_boundaries, "step1_boundaries")
        step1_vis_url = f"/temp/{os.path.basename(step1_vis_path)}"
        ctx.visualizations["step1_boundaries"] = step1_vis_url
        logger.info("Bubble Segmentation(Boundaries): %s", step1_vis_url)

    async def _run_text_detection(
        self, ctx: ProcessingContext, result: ImageResult
    ) -> None:
        """Execute text detection, rectangle refinement, cleaning, and outside text processing."""
        logger.info("═══════════════════════════ [STEP2] - BUBBLES TEXTS DETECTION AND RECTANGLE CALCULATED ═══════════════════════════")
        
        if not ctx.segments or not ctx.masks:
            logger.warning("[STEP2] Skipping text detection (no segments)")
            return

        # Build segment dicts with masks for text detection
        segments_with_mask = [
            {"id": seg.id, "box": seg.box, "mask": ctx.masks[i]}
            for i, seg in enumerate(ctx.segments)
            if i < len(ctx.masks)
        ]

        # Per-segment text boxes
        text_boxes_per_segment = await self.text_detection_service.detect_text_in_segments(
            ctx.image_rgb, segments_with_mask
        )
        ctx.text_boxes_per_segment = text_boxes_per_segment

        # Rectangle refinement
        refined_segments = self.segmentation_service.calculate_rectangles_with_text_boxes(
            ctx.segments, ctx.masks, text_boxes_per_segment, ctx.image_rgb
        )
        ctx.segments = refined_segments
        result.segments = refined_segments

        # Flatten rectangles metadata
        rectangles_flat: List[Dict[str, Any]] = []
        for seg in refined_segments:
            for rect_idx, rect in enumerate(seg.rectangles):
                x, y, w, h = rect
                rectangles_flat.append(
                    {
                        "segment_id": seg.id,
                        "rect_id": rect_idx,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                    }
                )
        ctx.rectangles_flat = rectangles_flat
        result.rectangles = rectangles_flat
        result.original_dimensions = (ctx.image_rgb.shape[1], ctx.image_rgb.shape[0])

        # Global text boxes
        all_text_boxes, all_text_scores = await self.text_detection_service.detect_all_text_boxes(
            ctx.image_rgb
        )
        ctx.all_text_boxes = all_text_boxes
        ctx.all_text_scores = all_text_scores

        # Clean text inside bubbles
        cleaned_image, blank_canvas_vis, text_vis = await self.text_detection_service.process_segments(
            ctx.image_rgb, segments_with_mask, text_boxes_per_segment
        )
        ctx.cleaned_image = cleaned_image

        # Save Step 2 visualizations
        step2_vis1_path = save_temp_image(blank_canvas_vis, "step2_blank_canvas")
        step2_vis2_path = save_temp_image(text_vis, "step2_text_masks")
        ctx.visualizations["step2_blank_canvas"] = f"/temp/{os.path.basename(step2_vis1_path)}"
        ctx.visualizations["step2_text_masks"] = f"/temp/{os.path.basename(step2_vis2_path)}"

        if cleaned_image is not None:
            cleaned_path = save_temp_image(cleaned_image, "cleaned_text")
            result.cleaned_text_result = f"/temp/{os.path.basename(cleaned_path)}"
            ctx.visualizations["cleaned_text"] = result.cleaned_text_result

        logger.info(
            "[STEP2] Visualization 1 (Blank canvas): %s",
            ctx.visualizations["step2_blank_canvas"],
        )
        logger.info(
            "[STEP2] Visualization 2 (Text masks + boxes): %s",
            ctx.visualizations["step2_text_masks"],
        )
        logger.info("[STEP2] Cleaned image: %s", result.cleaned_text_result)

        # Rectangles overlay visualization (after refinement)
        try:
            vis_rectangles = self.segmentation_service._create_step2_visualization3(
                ctx.image_rgb.copy(), refined_segments, ctx.masks
            )
            step2_vis3_path = save_temp_image(vis_rectangles, "step2_rectangles")
            ctx.visualizations["step2_rectangles"] = f"/temp/{os.path.basename(step2_vis3_path)}"
            logger.info(
                "[STEP2] Visualization 3 (Rectangles overlay): %s",
                ctx.visualizations["step2_rectangles"],
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[STEP2] Rectangles visualization failed: %s", e)

        # Text boxes OUTSIDE bubbles
        text_boxes_outside = self.text_detection_service.filter_boxes_outside_segments(
            all_text_boxes, all_text_scores, refined_segments
        )
        ctx.text_boxes_outside = text_boxes_outside
        logger.info(
            "[STEP2] Text boxes outside bubbles: %d",
            len(text_boxes_outside),
        )

        # Process outside bubble text (only if we have cleaned image)
        if len(text_boxes_outside) > 0 and cleaned_image is not None:
            outside_scores = all_text_scores[
                [
                    i
                    for i, box in enumerate(all_text_boxes)
                    if any(np.array_equal(box, ob) for ob in text_boxes_outside)
                ]
            ]
            (
                _outside_metadata,
                vis1_masks,
                vis2_black_canvas,
                vis3_filtered,
                vis4_filtered_masks,
                vis5_inpainted,
            ) = await self.text_detection_service.process_text_outside_bubbles(
                cleaned_image,
                text_boxes_outside,
                outside_scores,
                self.ocr_service,
                self.inpainting_service,
            )

            # Persist visualizations
            def _persist_vis(arr: np.ndarray, key: str) -> None:
                if arr is None:
                    return
                p = save_temp_image(arr, key)
                ctx.visualizations[key] = f"/temp/{os.path.basename(p)}"
                logger.info("[STEP2] %s: %s", key, ctx.visualizations[key])

            _persist_vis(vis1_masks, "text_outside_vis1_masks")
            _persist_vis(vis2_black_canvas, "text_outside_vis2_black_canvas")
            _persist_vis(vis3_filtered, "text_outside_vis3_filtered")
            _persist_vis(vis4_filtered_masks, "text_outside_vis4_filtered_masks")
            _persist_vis(vis5_inpainted, "text_outside_vis5_inpainted")

        logger.info("[STEP2] Completed ✓")

    async def _run_ocr(self, ctx: ProcessingContext, result: ImageResult) -> None:
        """Execute OCR over bubble segments."""
        logger.info("[STEP3] ═════════ OCR ═════════")
        if not ctx.segments:
            logger.info("[STEP3] No segments; skipping OCR")
            return

        ocr_results = await self.ocr_service.process_segments(ctx.image_rgb, ctx.segments)
        result.ocr_results = ocr_results
        logger.info("[STEP3] OCR results: %d segments", len(ocr_results))
        logger.info("[STEP3] Completed ✓")


__all__ = ["ImageProcessingPipelineService"]
