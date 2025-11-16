"""
Text Detection Service.

Responsibilities:
- Load ONNX-based comic text detector.
- Provide two granular detection modes:
    1. Global detection across entire image (detect_all_text_boxes)
    2. Per-segment detection restricted within segmentation masks (detect_text_in_segments)
- Clean text inside bubble segments (process_segments) producing:
    - Blank canvas with only bubble regions
    - Text removed (inpaint-like removal using OpenCV inpainting)
    - Visualizations of text masks and bounding boxes
- Process text outside bubble segments (process_text_outside_bubbles) with multi-stage filtering:
    - Morphological mask → candidate boxes
    - Merge / remove overlapping boxes
    - OCR-based semantic filtering & quality metrics
    - Optional inpainting of accepted masked text.

Clean Code / Refactor Notes:
- Centralized all tuning parameters into app.config.constants (imported as C).
- Added comprehensive type hints & docstrings for public methods.
- Normalized logging prefixes for easier log parsing.
- Reduced duplicate letterbox preprocessing logic (single _letterbox).
- Separated visualization creation helpers with clear naming.
- Added explicit return types & graceful fallbacks.
- Used early returns for error conditions; minimized nested conditionals.
- Ensured no business logic leaks into API schema layer.

Future Improvements (not yet implemented to keep behavior stable):
- Introduce a TextDetectionModel wrapper implementing BaseModel abstraction.
- Extract OCR filtering pipeline into dedicated strategy class.
- Convert heavy async OCR loops to use bounded concurrency for performance.

"""

import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageEnhance  # (retained if later visualization improvements)
from app.config import constants as C
from app.api.schemas.pipeline import SegmentData  # For type awareness (segments outside bubble filtering may use SegmentData)
from app.core.image_utils.geometry import filter_text_boxes_by_quality  # (placeholder import; adjust if actually needed)

logger = logging.getLogger(__name__)


class TextDetectionService:
    """
    High-level wrapper for comic text detection model.

    Args:
        model_base_path: Root path containing model subdirectories.

    Lifecycle:
        Instantiation triggers immediate ONNX model loading.
    """

    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.model = None
        self._load_model()

    # -------------------------------------------------------------------------
    # MODEL LOADING
    # -------------------------------------------------------------------------
    def _load_model(self) -> None:
        """
        Load ONNX text detection model.

        Raises:
            FileNotFoundError: if model file missing.
            RuntimeError: if model fails to load.
        """
        model_path = os.path.join(
            self.model_base_path, "text_detection", "comictextdetector.pt.onnx"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Text detection model not found: {model_path}")

        try:
            self.model = cv2.dnn.readNetFromONNX(model_path)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] Failed to load text detection model: {e}")
            raise RuntimeError(f"Failed to load text detection model: {e}") from e

    # -------------------------------------------------------------------------
    # PUBLIC PIPELINE ENTRYPOINTS
    # -------------------------------------------------------------------------
    async def process_segments(
        self,
        original_image: np.ndarray,
        segments: List[Dict[str, Any]],
        text_boxes_per_segment: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove text INSIDE bubble segments.

        Steps:
            1. Create blank canvas with ONLY balloon regions (masked copy).
            2. Run text detection on this canvas & remove text via inpainting approach.
            3. Produce visualizations of masks + boxes.

        Args:
            original_image: RGB image (H,W,3) uint8.
            segments: List of dict {id, box, mask} from segmentation phase.
            text_boxes_per_segment: Optional list of per-segment text boxes (global coords).

        Returns:
            (cleaned_image, blank_canvas_with_boundaries, text_vis_image)
        """
        self._assert_model_loaded()

        try:
            # Step 1: Construct blank canvas
            blank_canvas = self._build_blank_canvas(original_image, segments)
            blank_canvas_with_boundaries = self._draw_boundaries_on_canvas(
                blank_canvas.copy(), segments
            )

            # Step 2: Detect & remove text globally on blank canvas
            cleaned_canvas, text_mask, _global_boxes = self._remove_text_from_canvas(
                blank_canvas
            )

            # Step 3: Paste cleaned regions back into original image
            final_image = self._paste_cleaned_regions(
                original_image, cleaned_canvas, segments
            )

            # Visualization of text boxes per segment (if supplied)
            text_vis = self._create_step2_vis2(
                blank_canvas.copy(),
                segments,
                text_mask,
                text_boxes_per_segment or [],
            )

            return final_image, blank_canvas_with_boundaries, text_vis
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] process_segments error: {e}")
            raise

    async def detect_all_text_boxes(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ALL text boxes across entire original image.

        Args:
            image: RGB image.

        Returns:
            (boxes[N,4], scores[N]) both may be empty arrays if no detection.
        """
        self._assert_model_loaded()
        try:
            im_h, im_w = image.shape[:2]
            img_in, ratio, (dw, dh) = self._letterbox(
                image,
                new_shape=C.TEXT_DET_LETTERBOX_SIZE,
                stride=C.TEXT_DET_LETTERBOX_STRIDE,
            )

            blob = self._prepare_blob(img_in)
            outputs = self._forward(blob)
            boxes, _mask, scores = self._postprocess_text_detection(
                outputs, ratio, dw, dh, im_w, im_h
            )
            return boxes, scores
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] Global box detection error: {e}")
            return np.array([]), np.array([])

    async def detect_text_in_segments(
        self, image: np.ndarray, segments: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """
        Detect text boxes INSIDE each segmentation mask region.

        Args:
            image: RGB original image.
            segments: List[{'id','box','mask'}].

        Returns:
            List of arrays of shape (Mi, 4) each for global coordinates of boxes in segment i.
        """
        self._assert_model_loaded()

        results: List[np.ndarray] = []
        for idx, segment in enumerate(segments):
            x1, y1, x2, y2 = segment["box"]
            mask = segment["mask"]

            # Crop region
            cropped = image[y1:y2, x1:x2].copy()
            cropped_mask = mask[y1:y2, x1:x2]

            if cropped.size == 0:
                results.append(np.array([]))
                continue

            if cropped_mask.shape[:2] != cropped.shape[:2]:
                cropped_mask = cv2.resize(
                    cropped_mask,
                    (cropped.shape[1], cropped.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            masked_region = np.where(
                np.stack([cropped_mask] * 3, axis=-1) > 0, cropped, 0
            )

            img_in, ratio, (dw, dh) = self._letterbox(
                masked_region,
                new_shape=C.TEXT_DET_LETTERBOX_SIZE,
                stride=C.TEXT_DET_LETTERBOX_STRIDE,
            )
            blob = self._prepare_blob(img_in)
            outputs = self._forward(blob)
            local_boxes, _mask_local, _scores = self._postprocess_text_detection(
                outputs, ratio, dw, dh, cropped.shape[1], cropped.shape[0]
            )

            if len(local_boxes) > 0:
                global_boxes = local_boxes.copy()
                global_boxes[:, [0, 2]] += x1
                global_boxes[:, [1, 3]] += y1
                results.append(global_boxes)
            else:
                results.append(np.array([]))

        return results

    def filter_boxes_outside_segments(
        self,
        all_boxes: np.ndarray,
        all_scores: np.ndarray,
        segments: List[SegmentData],
        iou_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        Keep ONLY boxes that lie OUTSIDE segmentation regions.

        Args:
            all_boxes: (N,4) array [x1,y1,x2,y2]
            all_scores: (N,) scores
            segments: List[SegmentData]
            iou_threshold: IoU threshold (over box area) to treat as INSIDE a segment.

        Returns:
            Filtered boxes outside any segment area.
        """
        if len(all_boxes) == 0 or not segments:
            return all_boxes

        outside: List[List[int]] = []
        for box in all_boxes:
            bx1, by1, bx2, by2 = box
            box_area = (bx2 - bx1) * (by2 - by1)
            is_outside = True
            for seg in segments:
                sx1, sy1, sx2, sy2 = seg.box

                ix1 = max(bx1, sx1)
                iy1 = max(by1, sy1)
                ix2 = min(bx2, sx2)
                iy2 = min(by2, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    iou = inter / (box_area + 1e-6)
                    if iou > iou_threshold:
                        is_outside = False
                        break
            if is_outside:
                outside.append([bx1, by1, bx2, by2])

        result = np.array(outside) if outside else np.array([])
        return result

    async def process_text_outside_bubbles(
        self,
        image: np.ndarray,
        text_boxes_outside: np.ndarray,
        text_scores_outside: np.ndarray,
        ocr_service,
        inpainting_service=None,
        segments_data: Optional[List[SegmentData]] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process text OUTSIDE bubble segments, producing multi-step visualizations.

        Returns:
            (
                text_outside_metadata,
                vis1_masks,
                vis2_black_canvas,
                vis3_filtered_boxes,
                vis4_filtered_masks,
                vis5_calculated_rectangle,
                vis6_inpainted_result
            )
        """
        self._assert_model_loaded()
        try:
            im_h, im_w = image.shape[:2]
            img_in, ratio, (dw, dh) = self._letterbox(
                image,
                new_shape=C.TEXT_DET_LETTERBOX_SIZE,
                stride=C.TEXT_DET_LETTERBOX_STRIDE,
            )
            blob = self._prepare_blob(img_in)
            outputs = self._forward(blob)
            _, global_mask, _ = self._postprocess_text_detection(
                outputs, ratio, dw, dh, im_w, im_h
            )

            # Extract boxes from mask (stricter min area outside bubbles)
            boxes_from_mask = self._extract_boxes_from_mask(
                global_mask,
                min_area=C.TEXT_DET_BOX_MIN_AREA * 2,
                max_area=C.TEXT_DET_BOX_MAX_AREA,
                min_aspect_ratio=C.TEXT_DET_BOX_MIN_ASPECT,
                max_aspect_ratio=C.TEXT_DET_BOX_MAX_ASPECT,
                min_solidity=C.TEXT_DET_BOX_MIN_SOLIDITY + 0.1,
                merge_kernel_size=3,
            )

            if len(boxes_from_mask) > 0:
                boxes_from_mask = self._merge_nearby_boxes(
                    boxes_from_mask,
                    distance_threshold=C.TEXT_DET_MERGE_DISTANCE_THRESHOLD,
                    direction="both",
                )
                boxes_from_mask = self._filter_overlapping_boxes(
                    boxes_from_mask, iou_threshold=C.TEXT_DET_OVERLAP_IOU_THRESHOLD
                )

            # VIS 1: Masks over cleaned image (red overlay)
            vis1_masks = self._overlay_mask_red(image.copy(), global_mask)

            # VIS 2: Black canvas with red masks
            vis2_black_canvas = self._mask_on_black(global_mask, im_h, im_w)

            # VIS 3: OCR-based filtering & quality scoring
            vis3_filtered, filtered_indices = await self._create_vis3_filtered_boxes(
                vis2_black_canvas.copy(), boxes_from_mask, image, ocr_service
            )

            # VIS 4: Filtered masks only
            vis4_filtered_masks, filtered_mask = await self._create_vis4_filtered_masks(
                image.copy(), global_mask, boxes_from_mask, filtered_indices
            )


            # VIS 5: Inherit filtered boxes directly (no aggregation into single rectangle)
            accepted_boxes = (
                boxes_from_mask[filtered_indices] if filtered_indices and len(boxes_from_mask) > 0 else np.array([])
            )
            vis5_rectangle = image.copy()
            if len(accepted_boxes) > 0:
                for b in accepted_boxes:
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(vis5_rectangle, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(
                        vis5_rectangle,
                        "BOX",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
            # Build outside metadata (list of box dicts) for downstream final result overlay
            outside_metadata = [{"box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])]} for b in accepted_boxes] if len(accepted_boxes) > 0 else []

            # VIS 6: Optional inpainting (renamed from vis5_inpainted)
            if inpainting_service and filtered_mask is not None:
                vis6_inpainted = await inpainting_service.inpaint_text_regions(
                    image.copy(), filtered_mask, dilate_kernel_size=5
                )
            else:
                vis6_inpainted = image.copy()

            return (
                outside_metadata,
                vis1_masks,
                vis2_black_canvas,
                vis3_filtered,
                vis4_filtered_masks,
                vis5_rectangle,
                vis6_inpainted,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] process_text_outside_bubbles error: {e}")
            blank = image.copy()
            black = np.zeros_like(blank)
            return [], blank, black, black, blank, blank

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS: PREPROCESS / FORWARD / POSTPROCESS
    # -------------------------------------------------------------------------
    def _assert_model_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError("Text detection model not loaded")

    def _prepare_blob(self, img_in: np.ndarray) -> np.ndarray:
        """Convert letterboxed image to model blob."""
        blob = img_in.transpose((2, 0, 1))[::-1]
        blob = np.array([np.ascontiguousarray(blob)]).astype(np.float32) / 255.0
        return blob

    def _forward(self, blob: np.ndarray) -> List[np.ndarray]:
        """Run forward pass on prepared blob."""
        self.model.setInput(blob)
        output_names = self.model.getUnconnectedOutLayersNames()
        return self.model.forward(output_names)

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] | int = (1024, 1024),
        color: Tuple[int, int, int] = (0, 0, 0),
        stride: int = 64,
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Resize with unchanged aspect ratio using padding (letterbox).
        Returns:
            (letterboxed_image, (scale_h, scale_w), (pad_w, pad_h))
        """
        shape = img.shape[:2]  # (h, w)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dh, dw = int(dh), int(dw)

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
        return img, (r, r), (dw, dh)

    def _non_max_suppression_text(
        self,
        prediction: List[np.ndarray],
        conf_thres: float = C.TEXT_DET_NMS_CONF_THRESHOLD,
        iou_thres: float = C.TEXT_DET_NMS_IOU_THRESHOLD,
    ) -> List[Optional[np.ndarray]]:
        """Apply NMS over center-format predictions."""
        output: List[Optional[np.ndarray]] = [None] * len(prediction)
        for i, pred in enumerate(prediction):
            conf_mask = pred[:, 4] >= conf_thres
            pred = pred[conf_mask]
            if len(pred) == 0:
                continue

            boxes = pred[:, :4].copy()
            boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
            boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
            boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
            boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2

            scores = pred[:, 4]
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                indices = indices.flatten()
                output[i] = pred[indices]
        return output

    def _postprocess_text_detection(
        self,
        outputs: List[np.ndarray],
        ratio: Tuple[float, float],
        dw: int,
        dh: int,
        im_w: int,
        im_h: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess raw model outputs into (boxes, mask, scores).
        Boxes converted to corner format & scaled to original image dimensions.
        """
        blks = outputs[0]
        mask = outputs[1]
        lines_map = outputs[2]

        # Swap if model output ordering differs
        if mask.shape[1] == 2:
            mask, lines_map = lines_map, mask

        det = self._non_max_suppression_text(blks, C.TEXT_DET_NMS_CONF_THRESHOLD, C.TEXT_DET_NMS_IOU_THRESHOLD)[0]

        # Process mask
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        elif len(mask.shape) == 3:
            mask = mask[0]
        mask = (mask > C.TEXT_DET_MASK_BIN_THRESHOLD).astype(np.uint8) * 255

        # Remove padding & resize to original size
        mask = mask[: mask.shape[0] - int(dh), : mask.shape[1] - int(dw)]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        # Boxes
        boxes = np.array([])
        scores = np.array([])
        if det is not None and len(det) > 0:
            det_boxes = det[..., :4].copy()
            det_boxes[..., 0] = det[..., 0] - det[..., 2] / 2
            det_boxes[..., 1] = det[..., 1] - det[..., 3] / 2
            det_boxes[..., 2] = det[..., 0] + det[..., 2] / 2
            det_boxes[..., 3] = det[..., 1] + det[..., 3] / 2

            resize_ratio = (im_w / (C.TEXT_DET_LETTERBOX_SIZE - dw), im_h / (C.TEXT_DET_LETTERBOX_SIZE - dh))
            det_boxes[..., [0, 2]] *= resize_ratio[0]
            det_boxes[..., [1, 3]] *= resize_ratio[1]

            det_boxes[..., 0] = np.clip(det_boxes[..., 0], 0, im_w)
            det_boxes[..., 1] = np.clip(det_boxes[..., 1], 0, im_h)
            det_boxes[..., 2] = np.clip(det_boxes[..., 2], 0, im_w)
            det_boxes[..., 3] = np.clip(det_boxes[..., 3], 0, im_h)

            boxes = det_boxes.astype(np.int32)
            scores = np.round(det[..., 4], 3)

        return boxes, mask, scores

    # -------------------------------------------------------------------------
    # TEXT REMOVAL / CANVAS CLEANING
    # -------------------------------------------------------------------------
    def _remove_text_from_canvas(
        self, canvas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect text regions on a provided canvas and return a cleaned version
        with text removed (basic inpainting / blanking), along with the binary
        text mask and detected bounding boxes.

        Args:
            canvas: RGB image (H,W,3)

        Returns:
            (cleaned_canvas, text_mask, boxes_array)
        """
        self._assert_model_loaded()
        try:
            h, w = canvas.shape[:2]
            img_in, ratio, (dw, dh) = self._letterbox(
                canvas,
                new_shape=C.TEXT_DET_LETTERBOX_SIZE,
                stride=C.TEXT_DET_LETTERBOX_STRIDE,
            )
            blob = self._prepare_blob(img_in)
            outputs = self._forward(blob)
            boxes, mask, scores = self._postprocess_text_detection(
                outputs, ratio, dw, dh, w, h
            )

            if mask.size == 0:
                logger.debug("[TextDet] _remove_text_from_canvas: empty mask")
                return canvas, np.zeros((h, w), dtype=np.uint8), np.array([])

            cleaned = canvas.copy()
            if mask.shape[:2] != cleaned.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (cleaned.shape[1], cleaned.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            inpaint_mask = (mask > 0).astype(np.uint8) * 255
            try:
                # Use fast inpainting; fall back to white fill if it fails
                cleaned = cv2.inpaint(cleaned, inpaint_mask, 3, cv2.INPAINT_NS)
            except Exception as inpaint_err:  # noqa: BLE001
                logger.debug(
                    f"[TextDet] Inpainting failed, fallback white fill: {inpaint_err}"
                )
                cleaned[inpaint_mask > 0] = [255, 255, 255]

            return cleaned, mask, boxes
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _remove_text_from_canvas error: {e}")
            blank_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            return canvas, blank_mask, np.array([])

    # -------------------------------------------------------------------------
    # BOX EXTRACTION & MERGE / FILTER
    # -------------------------------------------------------------------------
    def _extract_boxes_from_mask(
        self,
        mask: np.ndarray,
        min_area: int = C.TEXT_DET_BOX_MIN_AREA,
        max_area: int = C.TEXT_DET_BOX_MAX_AREA,
        min_aspect_ratio: float = C.TEXT_DET_BOX_MIN_ASPECT,
        max_aspect_ratio: float = C.TEXT_DET_BOX_MAX_ASPECT,
        min_solidity: float = C.TEXT_DET_BOX_MIN_SOLIDITY,
        merge_kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Extract candidate boxes from binary mask using connected components & filtering constraints.
        """
        try:
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (merge_kernel_size, merge_kernel_size)
            )
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                mask_closed, connectivity=8
            )

            boxes: List[List[int]] = []
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                if area < min_area or area > max_area:
                    continue

                aspect = w / max(h, 1)
                if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
                    continue

                bbox_area = w * h
                solidity = area / max(bbox_area, 1)
                if solidity < min_solidity:
                    continue

                boxes.append([x, y, x + w, y + h])

            return np.array(boxes, dtype=np.int32) if boxes else np.array([])
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _extract_boxes_from_mask error: {e}")
            return np.array([])

    def _merge_nearby_boxes(
        self,
        boxes: np.ndarray,
        distance_threshold: int = C.TEXT_DET_MERGE_DISTANCE_THRESHOLD,
        direction: str = "both",
    ) -> np.ndarray:
        """
        Merge boxes that are spatially close (vertical/horizontal) with overlapping projection.
        """
        if len(boxes) == 0:
            return boxes

        try:
            merged: List[List[int]] = []
            used = set()

            for i in range(len(boxes)):
                if i in used:
                    continue

                current = boxes[i].copy()
                x1, y1, x2, y2 = current

                changed = True
                while changed:
                    changed = False
                    for j in range(len(boxes)):
                        if j == i or j in used:
                            continue
                        bx1, by1, bx2, by2 = boxes[j]

                        v_dist = min(abs(y1 - by2), abs(by1 - y2))
                        h_dist = min(abs(x1 - bx2), abs(bx1 - x2))
                        x_overlap = max(0, min(x2, bx2) - max(x1, bx1))
                        y_overlap = max(0, min(y2, by2) - max(y1, by1))

                        merge = False
                        if direction in ("vertical", "both"):
                            if v_dist <= distance_threshold and x_overlap > 0:
                                merge = True
                        if direction in ("horizontal", "both"):
                            if h_dist <= distance_threshold and y_overlap > 0:
                                merge = True

                        if merge:
                            x1 = min(x1, bx1)
                            y1 = min(y1, by1)
                            x2 = max(x2, bx2)
                            y2 = max(y2, by2)
                            current = [x1, y1, x2, y2]
                            used.add(j)
                            changed = True

                merged.append(current)
                used.add(i)

            return np.array(merged, dtype=np.int32) if merged else np.array([])
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _merge_nearby_boxes error: {e}")
            return boxes

    def _filter_overlapping_boxes(
        self, boxes: np.ndarray, iou_threshold: float = C.TEXT_DET_OVERLAP_IOU_THRESHOLD
    ) -> np.ndarray:
        """
        Remove boxes that have high overlap with larger boxes (IoU computed against smaller area).
        """
        if len(boxes) == 0:
            return boxes
        try:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            indices = np.argsort(areas)[::-1]

            kept: List[List[int]] = []
            for idx in indices:
                box_i = boxes[idx]
                x1_i, y1_i, x2_i, y2_i = box_i
                area_i = areas[idx]
                keep_flag = True
                for kb in kept:
                    ix1 = max(x1_i, kb[0])
                    iy1 = max(y1_i, kb[1])
                    ix2 = min(x2_i, kb[2])
                    iy2 = min(y2_i, kb[3])
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        iou = inter / min(area_i, (kb[2] - kb[0]) * (kb[3] - kb[1]))
                        if iou > iou_threshold:
                            keep_flag = False
                            break
                if keep_flag:
                    kept.append(box_i)

            return np.array(kept, dtype=np.int32) if kept else np.array([])
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _filter_overlapping_boxes error: {e}")
            return boxes

    # -------------------------------------------------------------------------
    # OCR-BASED FILTERING VISUALIZATION HELPERS
    # -------------------------------------------------------------------------
    async def _create_vis3_filtered_boxes(
        self,
        vis2_canvas: np.ndarray,
        boxes: np.ndarray,
        original_image: np.ndarray,
        ocr_service,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Apply OCR-based filtering pipeline for outside text detection.

        Returns:
            (visualization_canvas, accepted_box_original_indices)
        """
        if len(boxes) == 0:
            return vis2_canvas, []

        filtered: List[Tuple[np.ndarray, int]] = []
        rejected: List[Tuple[np.ndarray, int, str]] = []
        filtered_conf: List[Dict[str, Any]] = []

        BOX_MIN_SIZE = C.VIS3_BOX_MIN_SIZE
        OCR_MAX_SIZE = C.VIS3_OCR_MAX_SIZE
        LARGE_BOX_MIN_CHARS = C.VIS3_LARGE_BOX_MIN_CHARS
        MIN_TEXT_DENSITY = C.VIS3_MIN_TEXT_DENSITY
        BLACKLIST_PATTERN = C.VIS3_BLACKLIST_PATTERN

        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < BOX_MIN_SIZE:
                rejected.append((box, box_idx, "too_small"))
                continue

            if area >= OCR_MAX_SIZE:
                ar = w / max(h, 1)
                if 0.05 <= ar <= 30.0:
                    filtered.append((box, box_idx))
                    filtered_conf.append(
                        {
                            "confidence": 50.0,
                            "quality": "fair",
                            "metrics": {
                                "text_length": 0,
                                "char_density": 0.0,
                                "has_meaningful_chars": False,
                            },
                        }
                    )
                else:
                    rejected.append((box, box_idx, "bad_aspect_ratio"))
                continue

            if ocr_service.ocr_model is None:
                ar = w / max(h, 1)
                if 0.1 <= ar <= 20.0:
                    filtered.append((box, box_idx))
                    filtered_conf.append(
                        {
                            "confidence": 30.0,
                            "quality": "poor",
                            "metrics": {
                                "text_length": 0,
                                "char_density": 0.0,
                                "has_meaningful_chars": False,
                            },
                        }
                    )
                else:
                    rejected.append((box, box_idx, "bad_aspect_ratio"))
                continue

            # Crop with small padding
            pad = 2
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(original_image.shape[1], x2 + pad)
            y2c = min(original_image.shape[0], y2 + pad)

            if x2c <= x1c or y2c <= y1c:
                rejected.append((box, box_idx, "invalid_crop"))
                continue

            cropped = original_image[y1c:y2c, x1c:x2c]
            if cropped.size == 0:
                rejected.append((box, box_idx, "empty_crop"))
                continue

            # Image quality (simple laplacian variance / brightness / contrast checks)
            quality = self._calculate_image_quality(cropped)
            if not quality["is_acceptable"]:
                rejected.append((box, box_idx, "poor_quality"))
                continue

            try:
                text = await ocr_service._run_ocr(cropped)
            except Exception:
                text = "[OCR ERROR]"

            if text == "[OCR ERROR]" or not text:
                rejected.append((box, box_idx, "ocr_failed"))
                continue

            text_stripped = text.strip()
            if area > 20000 and len(text_stripped) < LARGE_BOX_MIN_CHARS:
                rejected.append((box, box_idx, "too_few_chars"))
                continue
            if len(text_stripped) < 5:
                rejected.append((box, box_idx, "text_too_short"))
                continue

            if not ocr_service._is_valid_text(text_stripped):
                rejected.append((box, box_idx, "invalid_text"))
                continue

            density = self._calculate_text_density(text_stripped, area)
            if density < MIN_TEXT_DENSITY:
                rejected.append((box, box_idx, "low_density"))
                continue

            ocr_quality = ocr_service.calculate_ocr_confidence(text_stripped, area)
            filtered.append((box, box_idx))
            filtered_conf.append(ocr_quality)

        # Build visualization
        canvas = vis2_canvas.copy()
        for idx, (box, original_idx) in enumerate(filtered):
            x1, y1, x2, y2 = box
            conf = filtered_conf[idx]["confidence"]
            qual = filtered_conf[idx]["quality"]
            color = (
                (0, 255, 0)
                if qual == "excellent"
                else (0, 255, 255)
                if qual == "good"
                else (0, 165, 255)
                if qual == "fair"
                else (0, 0, 255)
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"#{original_idx} {conf:.0f}%"
            self._draw_label(canvas, label, x1, y1, color)

        # Rejected (yellow)
        for box, original_idx, _reason in rejected:
            x1, y1, x2, y2 = box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            self._draw_label(canvas, f"#{original_idx} X", x1, y1, (0, 255, 255))

        accepted_indices = [orig_idx for _, orig_idx in filtered]
        return canvas, accepted_indices

    async def _create_vis4_filtered_masks(
        self,
        image: np.ndarray,
        full_mask: np.ndarray,
        all_boxes: np.ndarray,
        filtered_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create visualization of ONLY accepted mask regions from VIS3 results.
        Returns:
            (vis4_canvas, filtered_mask)
        """
        try:
            filtered_mask = np.zeros_like(full_mask, dtype=np.uint8)
            for idx in filtered_indices:
                if idx < len(all_boxes):
                    x1, y1, x2, y2 = all_boxes[idx]
                    filtered_mask[y1:y2, x1:x2] = np.maximum(
                        filtered_mask[y1:y2, x1:x2], full_mask[y1:y2, x1:x2]
                    )
            vis4 = self._overlay_mask_red(image.copy(), filtered_mask)
            return vis4, filtered_mask
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _create_vis4_filtered_masks error: {e}")
            return image, np.zeros_like(full_mask, dtype=np.uint8)

    # -------------------------------------------------------------------------
    # VISUALIZATION HELPERS
    # -------------------------------------------------------------------------
    def _build_blank_canvas(
        self, image: np.ndarray, segments: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Construct blank canvas with only segment regions pasted."""
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for seg in segments:
            x1, y1, x2, y2 = seg["box"]
            mask = seg["mask"]
            region = image[y1:y2, x1:x2].copy()
            seg_mask = mask[y1:y2, x1:x2]

            if seg_mask.shape[:2] != region.shape[:2]:
                seg_mask = cv2.resize(
                    seg_mask,
                    (region.shape[1], region.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_3ch = np.stack([seg_mask] * 3, axis=-1)
            canvas_roi = canvas[y1:y2, x1:x2]
            if canvas_roi.shape != region.shape:
                region = cv2.resize(
                    region,
                    (canvas_roi.shape[1], canvas_roi.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_3ch = cv2.resize(
                    mask_3ch,
                    (canvas_roi.shape[1], canvas_roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            canvas[y1:y2, x1:x2] = np.where(mask_3ch > 0, region, canvas_roi)
        return canvas

    def _draw_boundaries_on_canvas(
        self, canvas: np.ndarray, segments: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw green contours of segment masks on canvas."""
        vis = canvas.copy()
        for seg in segments:
            mask = seg["mask"]
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, (0, 255, 0), thickness=3)
        return vis

    def _create_step2_vis2(
        self,
        canvas: np.ndarray,
        segments: List[Dict[str, Any]],
        text_mask: np.ndarray,
        text_boxes_per_segment: List[np.ndarray],
    ) -> np.ndarray:
        """Visualization of text mask overlay + per-segment text boxes."""
        vis = self._overlay_mask_red(canvas.copy(), text_mask)
        for seg_idx, boxes in enumerate(text_boxes_per_segment):
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self._draw_label(vis, f"S{seg_idx}_B{box_idx}", x1, y1, (0, 255, 0))
        return vis

    def _paste_cleaned_regions(
        self,
        original_image: np.ndarray,
        cleaned_canvas: np.ndarray,
        segments: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Paste cleaned regions back to original image guided by masks."""
        final = original_image.copy()
        for seg in segments:
            x1, y1, x2, y2 = seg["box"]
            mask = seg["mask"]
            cleaned_region = cleaned_canvas[y1:y2, x1:x2].copy()
            cropped_mask = mask[y1:y2, x1:x2]
            if cropped_mask.shape[:2] != cleaned_region.shape[:2]:
                cropped_mask = cv2.resize(
                    cropped_mask,
                    (cleaned_region.shape[1], cleaned_region.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
            roi = final[y1:y2, x1:x2]
            if roi.shape[:2] == cleaned_region.shape[:2]:
                final[y1:y2, x1:x2] = np.where(mask_3ch > 0, cleaned_region, roi)
        return final

    def _overlay_mask_red(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply semi-transparent red overlay for given binary mask."""
        mask_colored = np.zeros_like(image)
        mask_colored[:, :] = [255, 0, 0]
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        return (image * (1 - mask_3ch * 0.5) + mask_colored * mask_3ch * 0.5).astype(
            np.uint8
        )

    def _mask_on_black(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Render mask on black background."""
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mask_colored = np.zeros_like(canvas)
        mask_colored[:, :] = [255, 0, 0]
        mask_3ch_full = np.stack([mask] * 3, axis=-1) / 255.0
        return (mask_colored * mask_3ch_full * 0.5).astype(np.uint8)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> None:
        """Draw label box + text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tx = x + 5
        ty = y + th + 5
        cv2.rectangle(
            image,
            (tx - 2, ty - th - 2),
            (tx + tw + 2, ty + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (tx, ty),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # TEXT / IMAGE QUALITY METRICS
    # -------------------------------------------------------------------------
    def _calculate_text_density(self, text: str, box_area: int) -> float:
        """Compute meaningful character density inside a box (chars per pixel)."""
        if not text or box_area <= 0:
            return 0.0
        meaningful = sum(
            1
            for ch in text
            if ch.isalpha()
            or "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u309f"
            or "\u30a0" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        )
        return meaningful / box_area

    def _calculate_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Compute basic quality metrics (blur, brightness, contrast) & acceptance flag.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_brightness = np.mean(gray)
            contrast = np.std(gray)

            if lap_var >= 500:
                sharp_score = 100
            elif lap_var >= 100:
                sharp_score = 50 + (lap_var - 100) / 400 * 50
            elif lap_var >= 50:
                sharp_score = 25 + (lap_var - 50) / 50 * 25
            else:
                sharp_score = lap_var / 50 * 25

            is_ok = (
                lap_var >= C.IMG_QUALITY_MIN_LAPLACIAN_VAR
                and contrast >= C.IMG_QUALITY_MIN_CONTRAST
                and C.IMG_QUALITY_MIN_BRIGHTNESS
                <= mean_brightness
                <= C.IMG_QUALITY_MAX_BRIGHTNESS
            )

            return {
                "variance": round(lap_var, 2),
                "mean_brightness": round(mean_brightness, 2),
                "contrast": round(contrast, 2),
                "sharpness_score": round(sharp_score, 2),
                "is_acceptable": is_ok,
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"[TextDet] _calculate_image_quality error: {e}")
            return {
                "variance": 0.0,
                "mean_brightness": 0.0,
                "contrast": 0.0,
                "sharpness_score": 0.0,
                "is_acceptable": False,
            }


__all__ = ["TextDetectionService"]
