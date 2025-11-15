"""
Segmentation service.

Responsibilities:
- Load ONNX segmentation model (bubble/balloon segmentation).
- Preprocess input image (letterbox with padding) and run inference.
- Postprocess raw model outputs: confidence filtering, NMS, mask generation.
- Provide rectangle calculation logic integrating downstream text boxes.
- Expose visualization helpers for pipeline steps.

Refactor Notes (clean code):
- Centralized all magic numbers into app.config.constants (imported as C).
- Added comprehensive type hints & docstrings.
- Normalized logging prefixes.
- Fixed _postprocess return type annotation (was incorrect).
- Removed duplicate hard-coded thresholds (using constants).
- Externalized expansion coverage thresholds & parameters to constants.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import List, Tuple, Optional

from app.api.schemas.pipeline import SegmentData
from app.core.image_utils.geometry import (
    find_largest_inscribed_rectangle,
    apply_nms,
    filter_segments_by_quality,
)
from app.config import constants as C

logger = logging.getLogger(__name__)


class SegmentationService:
    """
    High-level wrapper for segmentation model.

    Args:
        model_base_path: Root directory containing model subfolders.

    Lifecycle:
        Instantiate -> loads model session immediately (InferenceSession).
    """

    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load ONNX segmentation model.

        Raises:
            FileNotFoundError: If model path missing.
            RuntimeError: If inference session fails to initialize.
        """
        model_path = os.path.join(
            self.model_base_path, "segmentation", "manga_bubble_seg.onnx"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")

        try:
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            logger.info("[Segmentation] Model loaded")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[Segmentation] Failed to load model: {e}")
            raise RuntimeError(f"Failed to load segmentation model: {e}") from e

    async def process(
        self, image: np.ndarray
    ) -> Tuple[List[SegmentData], Optional[np.ndarray], List[np.ndarray]]:
        """
        Run segmentation end-to-end.

        Args:
            image: RGB image array (H,W,3) uint8.

        Returns:
            Tuple:
                - segments: List[SegmentData] (no rectangles yet)
                - visualization image (currently None placeholder)
                - masks: list of binary masks (H,W) uint8 for each segment
        """
        if self.session is None:
            raise RuntimeError("Segmentation model not loaded")

        try:
            original_h, original_w = image.shape[:2]

            # Preprocess
            input_tensor, scale, pad_w, pad_h = self._preprocess(image)

            # Inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})

            # Postprocess (returns segments + masks)
            segments, masks = self._postprocess(
                outputs, scale, pad_w, pad_h, original_w, original_h, image
            )

            # Quality filtering (std dev + area)
            if segments:
                filtered_segments, _quality_stats = filter_segments_by_quality(
                    segments,
                    std_threshold=C.SEGMENTATION_STD_THRESHOLD,
                    min_area=C.SEGMENTATION_MIN_AREA,
                )
                filtered_ids = {seg.id for seg in filtered_segments}
                filtered_masks = [
                    masks[i] for i in range(len(masks)) if i in filtered_ids
                ]
                segments = filtered_segments
                masks = filtered_masks

            return segments, None, masks

        except Exception as e:  # noqa: BLE001
            logger.error(f"[Segmentation] Processing error: {e}")
            raise

    def _preprocess(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Letterbox resize + normalization for model input.

        Returns:
            (batched_tensor, scale_factor, pad_w, pad_h)
        """
        input_size = C.SEGMENTATION_INPUT_SIZE
        orig_h, orig_w = image.shape[:2]

        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_w = (input_size - new_w) // 2
        pad_h = (input_size - new_h) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        normalized = canvas.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched, scale, pad_w, pad_h

    def _postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        pad_w: int,
        pad_h: int,
        original_w: int,
        original_h: int,
        original_image: np.ndarray,
    ) -> Tuple[List[SegmentData], List[np.ndarray]]:
        """
        Convert raw model outputs into segment data + masks.

        Args:
            outputs: Raw ONNX outputs
            scale: Resize scale applied during preprocessing
            pad_w/pad_h: Horizontal/vertical padding applied
            original_w/original_h: Original image dims
            original_image: Reference image for cropping

        Returns:
            (segments, masks)
        """
        boxes_output = outputs[0][0].T  # (N, 37)
        masks_output = outputs[1]  # (1, 32, 160, 160)

        boxes = boxes_output[:, :4]
        class_scores = boxes_output[:, 4:5]
        mask_coeffs = boxes_output[:, 5:]

        # Confidence filtering
        valid_mask = (class_scores.squeeze() >= C.SEGMENTATION_CONF_THRESHOLD)
        if not np.any(valid_mask):
            return [], []

        valid_boxes = boxes[valid_mask]
        valid_scores = class_scores[valid_mask]
        valid_mask_coeffs = mask_coeffs[valid_mask]

        # NMS
        selected_indices = apply_nms(
            valid_boxes, valid_scores.squeeze(), iou_threshold=C.SEGMENTATION_IOU_THRESHOLD
        )
        if len(selected_indices) == 0:
            return [], []

        final_boxes = valid_boxes[selected_indices]
        final_scores = valid_scores[selected_indices]
        final_mask_coeffs = valid_mask_coeffs[selected_indices]

        # Mask proto
        mask_protos = masks_output[0]  # (32, 160, 160)
        mask_protos_reshaped = mask_protos.reshape(32, -1)

        segments: List[SegmentData] = []
        masks: List[np.ndarray] = []

        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        for i, (box, score, coeffs) in enumerate(
            zip(final_boxes, final_scores, final_mask_coeffs)
        ):
            # Generate mask logits
            mask_logits = np.matmul(coeffs, mask_protos_reshaped)
            mask_logits = mask_logits.reshape(160, 160)
            mask_prob = 1 / (1 + np.exp(-mask_logits))

            x_center, y_center, width, height = box
            mask_h, mask_w = 160, 160
            x1_mask = int((x_center - width / 2) * mask_w / C.SEGMENTATION_INPUT_SIZE)
            y1_mask = int((y_center - height / 2) * mask_h / C.SEGMENTATION_INPUT_SIZE)
            x2_mask = int((x_center + width / 2) * mask_w / C.SEGMENTATION_INPUT_SIZE)
            y2_mask = int((y_center + height / 2) * mask_h / C.SEGMENTATION_INPUT_SIZE)

            x1_mask, y1_mask = max(0, x1_mask), max(0, y1_mask)
            x2_mask, y2_mask = min(mask_w, x2_mask), min(mask_h, y2_mask)

            mask_cropped = np.zeros((mask_h, mask_w), dtype=np.float32)
            if x2_mask > x1_mask and y2_mask > y1_mask:
                mask_cropped[y1_mask:y2_mask, x1_mask:x2_mask] = mask_prob[
                    y1_mask:y2_mask, x1_mask:x2_mask
                ]

            # Resize to model input size (640x640), remove padding, then to original size
            mask_resized = cv2.resize(
                mask_cropped, (C.SEGMENTATION_INPUT_SIZE, C.SEGMENTATION_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
            )
            mask_no_pad = mask_resized[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
            mask_original_size = cv2.resize(
                mask_no_pad, (original_w, original_h), interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (
                (mask_original_size > C.SEGMENTATION_MASK_THRESHOLD).astype(np.uint8)
            )

            contours, _ = cv2.findContours(
                mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            x1, y1, w, h = cv2.boundingRect(largest_contour)
            x2, y2 = x1 + w, y1 + h

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)

            # Sanity crop (unused but ensures viability)
            cropped_original = original_image[y1:y2, x1:x2]
            if cropped_original.size == 0:
                continue

            segments.append(
                SegmentData(
                    id=i,
                    box=[x1, y1, x2, y2],
                    score=float(score),
                    rectangles=[],  # Calculated later with text boxes
                )
            )
            masks.append(mask_binary)

        return segments, masks

    def calculate_rectangles_with_text_boxes(
        self,
        segments: List[SegmentData],
        masks: List[np.ndarray],
        text_boxes_per_segment: List[np.ndarray],
        original_image: np.ndarray,
    ) -> List[SegmentData]:
        """
        Derive refined rectangles for each segment using associated detected text boxes.

        Logic:
            - If ≥ 2 text boxes: create one rectangle per text box, then expand with constraints.
            - If 1 or 0 text boxes: use largest inscribed rectangle algorithm within mask.

        Returns:
            Updated list of SegmentData with rectangles populated.
        """
        logger.info(
            f"[Rectangles] Calculating rectangles for {len(segments)} segments"
        )
        updated_segments: List[SegmentData] = []

        for idx, seg in enumerate(segments):
            x1, y1, x2, y2 = seg.box
            mask = masks[idx] if idx < len(masks) else None
            text_boxes = (
                text_boxes_per_segment[idx]
                if idx < len(text_boxes_per_segment)
                else np.array([])
            )

            num_boxes = len(text_boxes)
            rectangles: List[List[int]] = []

            if num_boxes >= 2:
                logger.info(
                    f"[Rectangles] Segment #{seg.id}: Multiple text boxes -> {num_boxes} initial rectangles"
                )
                for tb_idx, tb in enumerate(text_boxes):
                    tb_x1, tb_y1, tb_x2, tb_y2 = tb
                    tb_w = tb_x2 - tb_x1
                    tb_h = tb_y2 - tb_y1

                    rect_x = max(x1, tb_x1)
                    rect_y = max(y1, tb_y1)
                    rect_w = min(x2 - rect_x, tb_w)
                    rect_h = min(y2 - rect_y, tb_h)

                    if rect_w > 0 and rect_h > 0:
                        rectangles.append([rect_x, rect_y, rect_w, rect_h])

                if len(rectangles) >= 2:
                    rectangles = self._expand_rectangles_with_constraints(
                        rectangles=rectangles,
                        segment_box=seg.box,
                        mask=mask,
                        max_padding=C.RECT_EXPAND_MAX_PADDING,
                        padding_step=C.RECT_EXPAND_PADDING_STEP,
                        min_gap=C.RECT_EXPAND_MIN_GAP,
                    )
            else:
                # Fallback geometry approach
                if mask is not None:
                    cropped_mask = mask[y1:y2, x1:x2]
                    if cropped_mask.size > 0:
                        local_rect = find_largest_inscribed_rectangle(cropped_mask)
                        if local_rect is not None:
                            rect_x, rect_y, rect_w, rect_h = local_rect
                            rectangles.append(
                                [x1 + rect_x, y1 + rect_y, rect_w, rect_h]
                            )

            updated_segments.append(
                SegmentData(
                    id=seg.id, box=seg.box, score=seg.score, rectangles=rectangles
                )
            )

        logger.info("[Rectangles] Completed rectangle calculation")
        return updated_segments

    def _expand_rectangles_with_constraints(
        self,
        rectangles: List[List[int]],
        segment_box: List[int],
        mask: np.ndarray,
        max_padding: int,
        padding_step: int,
        min_gap: int,
    ) -> List[List[int]]:
        """
        Expand rectangles outward constrained by:
            - Segment bounds
            - Minimum gap between rectangles
            - Mask coverage constraints (edge + full coverage)

        Returns:
            Expanded rectangles in [x, y, w, h] format.
        """
        if len(rectangles) == 0:
            return rectangles

        seg_x1, seg_y1, seg_x2, seg_y2 = segment_box

        # Convert to [x1,y1,x2,y2]
        expanded = [[r[0], r[1], r[0] + r[2], r[1] + r[3]] for r in rectangles]

        for _padding in range(padding_step, max_padding + 1, padding_step):
            any_expanded = False

            for i in range(len(expanded)):
                x1, y1, x2, y2 = expanded[i]

                new_x1 = max(seg_x1, x1 - padding_step)
                new_y1 = max(seg_y1, y1 - padding_step)
                new_x2 = min(seg_x2, x2 + padding_step)
                new_y2 = min(seg_y2, y2 + padding_step)

                if new_x2 <= new_x1 or new_y2 <= new_y1:
                    continue

                # Collision check
                collision = False
                for j, other in enumerate(expanded):
                    if j == i:
                        continue
                    ox1, oy1, ox2, oy2 = other
                    if not (
                        new_x2 + min_gap < ox1
                        or new_x1 > ox2 + min_gap
                        or new_y2 + min_gap < oy1
                        or new_y1 > oy2 + min_gap
                    ):
                        collision = True
                        break
                if collision:
                    continue

                img_h, img_w = mask.shape[:2]
                cx1 = max(0, min(img_w - 1, new_x1))
                cy1 = max(0, min(img_h - 1, new_y1))
                cx2 = max(0, min(img_w, new_x2))
                cy2 = max(0, min(img_h, new_y2))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue

                sub_mask = mask[cy1:cy2, cx1:cx2]
                if sub_mask.size == 0:
                    continue

                # Edge coverage
                def edge_coverage(line: np.ndarray) -> float:
                    if line.size == 0:
                    # avoid division by zero
                        return 0.0
                    return float(np.sum(line > 0) / line.size)

                top_cov = edge_coverage(sub_mask[0, :])
                bottom_cov = edge_coverage(sub_mask[-1, :])
                left_cov = edge_coverage(sub_mask[:, 0])
                right_cov = edge_coverage(sub_mask[:, -1])

                if (
                    top_cov < C.RECT_EXPAND_EDGE_MIN_COVERAGE
                    or bottom_cov < C.RECT_EXPAND_EDGE_MIN_COVERAGE
                    or left_cov < C.RECT_EXPAND_EDGE_MIN_COVERAGE
                    or right_cov < C.RECT_EXPAND_EDGE_MIN_COVERAGE
                ):
                    continue

                # Region coverage
                region_cov = float(np.sum(sub_mask > 0) / (sub_mask.size + 1e-6))
                if region_cov < C.RECT_EXPAND_REGION_MIN_COVERAGE:
                    continue

                expanded[i] = [new_x1, new_y1, new_x2, new_y2]
                any_expanded = True

            if not any_expanded:
                break

        # Back to [x,y,w,h]
        result: List[List[int]] = []
        for original, rect in zip(rectangles, expanded):
            x1, y1, x2, y2 = rect
            w = x2 - x1
            h = y2 - y1
            result.append([x1, y1, w, h])

        return result

    def _create_step1_visualization(
        self, image: np.ndarray, segments: List[SegmentData], masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Visualization helper for Step 1:
        Draw green boundaries (contours) of segment masks on original image.
        """
        vis = image.copy()
        for i, _seg in enumerate(segments):
            if i < len(masks):
                mask = masks[i]
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, (0, 255, 0), thickness=3)
        return vis

    def _create_step2_visualization3(
        self, image: np.ndarray, segments: List[SegmentData], masks: List[np.ndarray]
    ) -> np.ndarray:
        """
        Visualization helper for Step 2 (variant 3):
        Draw all rectangles (red) for each segment on original image.
        """
        vis = image.copy()
        for segment in segments:
            for rect_idx, rectangle in enumerate(segment.rectangles):
                x, y, w, h = rectangle
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                label = (
                    f"S{segment.id}_R{rect_idx}"
                    if len(segment.rectangles) > 1
                    else f"S{segment.id}"
                )
                (tw, th), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                text_x = x + 5
                text_y = y + th + 10
                cv2.rectangle(
                    vis,
                    (text_x - 2, text_y - th - 2),
                    (text_x + tw + 2, text_y + baseline + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 0, 0),
                    font_thickness,
                    cv2.LINE_AA,
                )
        return vis
