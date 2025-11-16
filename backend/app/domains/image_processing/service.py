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
        outside_final_boxes: Accepted outside boxes metadata (after filtering) for combined final result & OCR.
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
    outside_final_boxes: List[Dict[str, Any]] = field(default_factory=list)


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
        logger.info("═════════════════ [STEP1] - BUBBLES SEGMENTATION ═════════════════")
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
        logger.info("[STEP] Bubble Segmentation(Boundaries): %s", step1_vis_url)

    async def _run_text_detection(
        self, ctx: ProcessingContext, result: ImageResult
    ) -> None:
        """Execute text detection, rectangle refinement, cleaning, and outside text processing."""
        logger.info("═════════════════ [STEP2] - BUBBLES TEXTS DETECTION AND RECTANGLE CALCULATED ═════════════════")
        
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
            "[STEP2] [INSIDE BUBBLE] Use Black Canvas (Use From Step1) %s",
            ctx.visualizations["step2_blank_canvas"],
        )
        logger.info(
            "[STEP2] [INSIDE BUBBLE] Apply Text Masks And Caculated Boxes: %s",
            ctx.visualizations["step2_text_masks"],
        )
        logger.info("[STEP2] [INSIDE BUBBLE] Clean Texts And Return To Background: %s", result.cleaned_text_result)

        # Rectangles overlay visualization (after refinement)
        try:
            vis_rectangles = self.segmentation_service._create_step2_visualization3(
                ctx.image_rgb.copy(), refined_segments, ctx.masks
            )
            step2_vis3_path = save_temp_image(vis_rectangles, "step2_rectangles")
            ctx.visualizations["step2_rectangles"] = f"/temp/{os.path.basename(step2_vis3_path)}"
            logger.info(
                "[STEP2] [INSIDE BUBBLE] Final Calculated Rectangle: %s",
                ctx.visualizations["step2_rectangles"],
            )
            # Generate combined INSIDE BUBBLE final result (cleaned text + rectangles)
            try:
                if cleaned_image is not None:
                    inside_final = cleaned_image.copy()
                    for seg in refined_segments:
                        for rect in getattr(seg, "rectangles", []):
                            if len(rect) == 4:
                                x, y, w, h = rect
                                cv2.rectangle(
                                    inside_final,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0),
                                    3,
                                )
                                cv2.putText(
                                    inside_final,
                                    "RECT",
                                    (x, max(0, y - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )
                    inside_final_path = save_temp_image(inside_final, "inside_final_result")
                    ctx.visualizations["inside_final_result"] = f"/temp/{os.path.basename(inside_final_path)}"
                    logger.info(
                        "[STEP2] [INSIDE BUBBLE] Final Result: %s",
                        ctx.visualizations["inside_final_result"],
                    )
            except Exception as e_inside:  # noqa: BLE001
                logger.error("[STEP2] [INSIDE BUBBLE] Final Result generation failed: %s", e_inside)
        except Exception as e:  # noqa: BLE001
            logger.error("[STEP2] Rectangles visualization failed: %s", e)

        # Text boxes OUTSIDE bubbles
        text_boxes_outside = self.text_detection_service.filter_boxes_outside_segments(
            all_text_boxes, all_text_scores, refined_segments
        )
        ctx.text_boxes_outside = text_boxes_outside

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
                outside_metadata,
                vis1_masks,
                vis2_black_canvas,
                vis3_filtered,
                vis4_filtered_masks,
                vis5_rectangle,
                vis6_inpainted,
            ) = await self.text_detection_service.process_text_outside_bubbles(
                cleaned_image,
                text_boxes_outside,
                outside_scores,
                self.ocr_service,
                self.inpainting_service,
            )

                        # Persist visualizations
            def _persist_vis(arr: np.ndarray, file_prefix: str, display_name: str) -> None:
                if arr is None:
                    return
                p = save_temp_image(arr, file_prefix)
                ctx.visualizations[file_prefix] = f"/temp/{os.path.basename(p)}"
                logger.info("[STEP2] %s: %s", display_name, ctx.visualizations[file_prefix])

            _persist_vis(vis1_masks, "outside_text_masks", "[OUTSIDE] Apply Text Masks")
            _persist_vis(vis2_black_canvas, "outside_black_canvas", "[OUTSIDE] Use Black Canvas")
            _persist_vis(vis3_filtered, "outside_filtered_boxes", "[OUTSIDE] Apply Filter To Add Boxes Filter")
            _persist_vis(vis4_filtered_masks, "outside_filtered_masks", "[OUTSIDE] Remove Text Mask Filter Boxes And Return To Background")
            _persist_vis(vis5_rectangle, "outside_calculated_rectangle", "[OUTSIDE] Calculated Rectangle")
            _persist_vis(vis6_inpainted, "outside_inpainted", "[OUTSIDE] Clean Text Mask Using InPainting")

            # Generate combined OUTSIDE final result (inpainted + rectangles)
            try:
                if vis6_inpainted is not None:
                    outside_final = vis6_inpainted.copy()
                    # Draw rectangles from metadata if available
                    for item in outside_metadata or []:
                        box = item.get("box")
                        if box and len(box) == 4:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(
                                outside_final,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                3,
                            )
                            cv2.putText(
                                outside_final,
                                "BOX",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                    outside_final_path = save_temp_image(outside_final, "outside_final_result")
                    ctx.visualizations["outside_final_result"] = f"/temp/{os.path.basename(outside_final_path)}"
                    logger.info(
                        "[STEP2] [OUTSIDE] Final Result: %s",
                        ctx.visualizations["outside_final_result"],
                    )
                    # Persist accepted outside boxes metadata into context
                    ctx.outside_final_boxes = outside_metadata or []
                    # Generate combined INSIDE & OUTSIDE final result with numbering
                    try:
                        base_combined = vis6_inpainted.copy() if vis6_inpainted is not None else ctx.image_rgb.copy()
                        # Collect inside rectangles (convert (x,y,w,h) -> (x1,y1,x2,y2))
                        inside_rects = []
                        for seg in refined_segments:
                            for rect in getattr(seg, "rectangles", []):
                                if len(rect) == 4:
                                    x, y, w, h = rect
                                    inside_rects.append([x, y, x + w, y + h])
                        # Sort inside: top-to-bottom then left-to-right
                        inside_rects_sorted = sorted(inside_rects, key=lambda r: (r[1], r[0]))
                        # Outside boxes already [x1,y1,x2,y2]; sort similarly
                        outside_boxes_sorted = sorted(
                            [b["box"] for b in ctx.outside_final_boxes] if ctx.outside_final_boxes else [],
                            key=lambda r: (r[1], r[0]),
                        )
                        # Numbering: inside first then outside
                        counter = 1
                        def _draw_numbered(box, color):
                            nonlocal counter
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(base_combined, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(
                                base_combined,
                                f"{counter}",
                                (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                color,
                                2,
                                cv2.LINE_AA,
                            )
                            counter += 1
                        for box in inside_rects_sorted:
                            _draw_numbered(box, (0, 255, 0))  # Green for inside
                        for box in outside_boxes_sorted:
                            _draw_numbered(box, (0, 165, 255))  # Orange for outside
                        combined_path = save_temp_image(base_combined, "inside_outside_final_result")
                        ctx.visualizations["inside_outside_final_result"] = f"/temp/{os.path.basename(combined_path)}"
                        logger.info(
                            "[STEP2] [INSIDE&OUTSIDE] Final Result: %s",
                            ctx.visualizations["inside_outside_final_result"],
                        )
                    except Exception as e_comb:  # noqa: BLE001
                        logger.error("[STEP2] [INSIDE&OUTSIDE] Final Result generation failed: %s", e_comb)
            except Exception as e_outside:  # noqa: BLE001
                logger.error("[STEP2] [OUTSIDE] Final Result generation failed: %s", e_outside)

    async def _run_ocr(self, ctx: ProcessingContext, result: ImageResult) -> None:
        """Execute OCR over bubble segments."""
        logger.info("═════════════════ [STEP3] OCR ═════════════════")
        if not ctx.segments:
            logger.info("[STEP3] No segments; skipping OCR")
            return

        ocr_results = await self.ocr_service.process_segments(ctx.image_rgb, ctx.segments)
        result.ocr_results = ocr_results

        # Log inside OCR texts
        inside_texts = [res.original_text for res in ocr_results]
        if inside_texts:
            logger.info("[STEP3] OCR Inside Texts: %s", inside_texts)
        else:
            logger.info("[STEP3] OCR Inside Texts: []")

        # Log outside OCR texts (run OCR over accepted outside boxes if any)
        try:
            if ctx.outside_final_boxes:
                outside_boxes_arr = np.array([b["box"] for b in ctx.outside_final_boxes], dtype=np.int32)
                verified_boxes, outside_texts, _outside_conf = await self.ocr_service.verify_text_boxes(
                    ctx.image_rgb, outside_boxes_arr
                )
                logger.info("[STEP3] OCR Outside Texts: %s", outside_texts if outside_texts else [])
            else:
                logger.info("[STEP3] OCR Outside Texts: []")
        except Exception as e:  # noqa: BLE001
            logger.error("[STEP3] OCR Outside Texts collection failed: %s", e)
            logger.info("[STEP3] OCR Outside Texts: []")


__all__ = ["ImageProcessingPipelineService"]
