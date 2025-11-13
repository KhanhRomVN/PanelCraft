# app/services/text_detection_service.py
import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging
from app.utils.geometry_utils import filter_text_boxes_by_quality
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from pathlib import Path

logger = logging.getLogger(__name__)

class TextDetectionService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load text detection model"""
        model_path = os.path.join(self.model_base_path, "text_detection", "comictextdetector.pt.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Text detection model not found: {model_path}")
        
        try:
            self.model = cv2.dnn.readNetFromONNX(model_path)
            logger.info("Text detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text detection model: {e}")
            raise
    
    async def process_segments(
        self, 
        original_image: np.ndarray, 
        segments: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process segments để remove text và trả về ảnh gốc đã clean + visualizations
        
        Args:
            original_image: Ảnh gốc (RGB)
            segments: List segments từ segmentation (dạng dict với keys: id, box, mask)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (cleaned_image, blank_canvas_with_boundaries, text_vis)
        """
        if self.model is None:
            raise RuntimeError("Text detection model not loaded")
        
        try:
            # BƯỚC 1: TẠO BLANK CANVAS VỚI CHỈ CÓ BALLOON REGIONS
            original_h, original_w = original_image.shape[:2]
            blank_canvas = np.full((original_h, original_w, 3), 0, dtype=np.uint8)
            
            # Paste CHỈ balloon regions vào blank canvas
            for idx, segment in enumerate(segments):
                x1, y1, x2, y2 = segment['box']
                mask = segment['mask']
                
                cropped_region = original_image[y1:y2, x1:x2].copy()
                cropped_mask = mask[y1:y2, x1:x2]
                
                # Resize mask nếu cần
                if cropped_mask.shape[:2] != cropped_region.shape[:2]:
                    cropped_mask = cv2.resize(cropped_mask, 
                                            (cropped_region.shape[1], cropped_region.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                
                mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                canvas_roi = blank_canvas[y1:y2, x1:x2]
                
                if canvas_roi.shape != cropped_region.shape:
                    cropped_region = cv2.resize(cropped_region, 
                                               (canvas_roi.shape[1], canvas_roi.shape[0]), 
                                               interpolation=cv2.INTER_LINEAR)
                    mask_3ch = cv2.resize(mask_3ch, 
                                         (canvas_roi.shape[1], canvas_roi.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                
                blank_canvas[y1:y2, x1:x2] = np.where(mask_3ch > 0, cropped_region, canvas_roi)
            
            # Lưu blank canvas với green boundaries cho Step 2 Vis 1
            blank_canvas_with_boundaries = self._draw_boundaries_on_canvas(blank_canvas.copy(), segments)
            
            # BƯỚC 2: CHẠY TEXT DETECTION TRÊN BLANK CANVAS
            cleaned_canvas, text_mask, text_boxes = self._remove_text_from_canvas(blank_canvas)
            
            # BƯỚC 3: PASTE CLEANED SEGMENTS BACK TO ORIGINAL
            final_image = original_image.copy()
            
            for idx, segment in enumerate(segments):
                x1, y1, x2, y2 = segment['box']
                mask = segment['mask']
                
                cleaned_region = cleaned_canvas[y1:y2, x1:x2].copy()
                cropped_mask = mask[y1:y2, x1:x2]
                
                if cropped_mask.shape[:2] != cleaned_region.shape[:2]:
                    cropped_mask = cv2.resize(cropped_mask, 
                                            (cleaned_region.shape[1], cleaned_region.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
                
                mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                roi = final_image[y1:y2, x1:x2]
                
                if roi.shape[:2] == cleaned_region.shape[:2]:
                    final_image[y1:y2, x1:x2] = np.where(mask_3ch > 0, cleaned_region, roi)
            
            # Tạo Step 2 Vis 2: Text masks + boxes trên từng segment
            text_vis = self._create_step2_vis2(blank_canvas.copy(), segments, text_mask, text_boxes)
            
            return final_image, blank_canvas_with_boundaries, text_vis
            
        except Exception as e:
            logger.error(f"Text detection processing error: {e}")
            raise
    
    def _remove_text_from_canvas(
        self, 
        canvas: np.ndarray
    ) -> tuple:
        """
        Remove text từ toàn bộ blank canvas
        
        Args:
            canvas: Blank canvas chỉ chứa balloon regions (RGB)
        
        Returns:
            tuple: (cleaned_canvas, text_mask, text_boxes)
        """
        im_h, im_w = canvas.shape[:2]
        
        # Preprocess
        img_in, ratio, (dw, dh) = self._letterbox(canvas, new_shape=1024, stride=64)
        
        # Convert to blob
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        
        # Run inference
        self.model.setInput(img_in)
        output_names = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(output_names)
        
        # Postprocess
        boxes, mask, scores = self._postprocess_text_detection(
            outputs, ratio, dw, dh, im_w, im_h
        )
        
        # Inpaint toàn bộ canvas
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(canvas_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        
        return cleaned_rgb, mask, boxes
    
    def _draw_boundaries_on_canvas(self, canvas: np.ndarray, segments: List[dict]) -> np.ndarray:
        """
        Vẽ green boundaries lên blank canvas cho Step 2 Vis 1
        """
        vis_canvas = canvas.copy()
        
        for segment in segments:
            mask = segment['mask']
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_canvas, contours, -1, (0, 255, 0), thickness=3)
        
        return vis_canvas
    
    def _create_step2_vis2(
        self, 
        canvas: np.ndarray, 
        segments: List[dict],
        text_mask: np.ndarray,
        text_boxes: np.ndarray
    ) -> np.ndarray:
        """
        Tạo Step 2 Vis 2: Text masks + bounding boxes trên từng segment
        """
        vis_image = canvas.copy()
        
        # Bôi màu đỏ (0.5 alpha) lên text masks
        mask_colored = np.zeros_like(vis_image)
        mask_colored[:, :] = [255, 0, 0]
        
        mask_3ch = np.stack([text_mask] * 3, axis=-1) / 255.0
        vis_image = (vis_image * (1 - mask_3ch * 0.5) + mask_colored * mask_3ch * 0.5).astype(np.uint8)
        
        # Vẽ green bounding boxes
        if len(text_boxes) > 0:
            for box in text_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        
        return vis_image
    
    async def detect_all_text_boxes(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ALL text boxes trên toàn bộ ảnh gốc
        
        Args:
            image: Ảnh gốc (RGB)
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (boxes, scores)
        """
        if self.model is None:
            raise RuntimeError("Text detection model not loaded")
        
        try:
            im_h, im_w = image.shape[:2]
            
            # Preprocess
            img_in, ratio, (dw, dh) = self._letterbox(image, new_shape=1024, stride=64)
            
            # Convert to blob
            img_in = img_in.transpose((2, 0, 1))[::-1]
            img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
            
            # Run inference
            self.model.setInput(img_in)
            output_names = self.model.getUnconnectedOutLayersNames()
            outputs = self.model.forward(output_names)
            
            # Postprocess
            boxes, text_mask, scores = self._postprocess_text_detection(
                outputs, ratio, dw, dh, im_w, im_h
            )
            
            return boxes, scores
            
        except Exception as e:
            logger.error(f"[STEP2] Error detecting text boxes: {e}")
            return np.array([]), np.array([])
    
    def filter_boxes_outside_segments(
        self,
        all_boxes: np.ndarray,
        all_scores: np.ndarray,
        segments: List,
        iou_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Filter để giữ lại CHỈ các boxes NGOÀI bubble segments
        
        Args:
            all_boxes: Tất cả text boxes detected
            all_scores: Scores tương ứng
            segments: List segments từ segmentation
            iou_threshold: Ngưỡng IoU để coi là overlap (default: 0.3)
        
        Returns:
            np.ndarray: Boxes ngoài bubbles
        """
        if len(all_boxes) == 0 or len(segments) == 0:
            return all_boxes
        
        outside_boxes = []
        outside_scores = []
        
        for box, score in zip(all_boxes, all_scores):
            bx1, by1, bx2, by2 = box
            
            is_outside = True
            
            # Check IoU với tất cả segments
            for seg in segments:
                seg_x1, seg_y1, seg_x2, seg_y2 = seg.box
                
                # Calculate intersection
                ix1 = max(bx1, seg_x1)
                iy1 = max(by1, seg_y1)
                ix2 = min(bx2, seg_x2)
                iy2 = min(by2, seg_y2)
                
                if ix2 > ix1 and iy2 > iy1:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    
                    # Calculate box area
                    box_area = (bx2 - bx1) * (by2 - by1)
                    
                    # Calculate IoU (intersection over box area)
                    iou = intersection / (box_area + 1e-6)
                    
                    # Nếu overlap > threshold → box nằm TRONG segment
                    if iou > iou_threshold:
                        is_outside = False
                        break
            
            if is_outside:
                outside_boxes.append(box)
                outside_scores.append(score)
        
        outside_boxes = np.array(outside_boxes) if outside_boxes else np.array([])
        outside_scores = np.array(outside_scores) if outside_scores else np.array([])
        
        logger.info(f"[STEP2] Filtered {len(all_boxes)} -> {len(outside_boxes)} boxes outside (IoU threshold: {iou_threshold})")
        
        return outside_boxes
    
    async def process_text_outside_bubbles(
        self,
        image: np.ndarray,
        text_boxes_outside: np.ndarray,
        text_scores_outside: np.ndarray,
        ocr_service,
        segments_data: List = None
    ) -> Tuple[List[dict], dict]:
        """
        Xử lý text NGOÀI bubble: chạy text detection trên cleaned image và vẽ masks
        
        Args:
            image: Cleaned image (RGB)
            text_boxes_outside: KHÔNG SỬ DỤNG (giữ lại để tương thích API)
            text_scores_outside: KHÔNG SỬ DỤNG
            ocr_service: OCR service để verify text quality
        
        Returns:
            Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray]: (empty_data, vis1_masks, vis2_black_canvas, vis3_filtered)
        """
        try:
            im_h, im_w = image.shape[:2]
            
            # CHẠY TEXT DETECTION TRỰC TIẾP TRÊN CLEANED IMAGE
            img_in, ratio, (dw, dh) = self._letterbox(image, new_shape=1024, stride=64)
            
            img_in = img_in.transpose((2, 0, 1))[::-1]
            img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
            
            self.model.setInput(img_in)
            output_names = self.model.getUnconnectedOutLayersNames()
            outputs = self.model.forward(output_names)
            
            # Lấy boxes và mask từ model
            _, mask, _ = self._postprocess_text_detection(
                outputs, ratio, dw, dh, im_w, im_h
            )
            
            boxes_from_mask = self._extract_boxes_from_mask(
                mask,
                min_area=200,
                max_area=50000,
                min_aspect_ratio=0.1,
                max_aspect_ratio=20.0,
                min_solidity=0.4,
                merge_kernel_size=3
            )
            
            logger.info(f"[Text Outside] Detected {len(boxes_from_mask)} text boxes from masks")
            
            # BƯỚC MỚI: Merge boxes gần nhau
            if len(boxes_from_mask) > 0:
                boxes_from_mask = self._merge_nearby_boxes(
                    boxes_from_mask,
                    distance_threshold=50,
                    direction="both"
                )
            
            # BƯỚC MỚI: Filter boxes overlap
            if len(boxes_from_mask) > 0:
                boxes_from_mask = self._filter_overlapping_boxes(
                    boxes_from_mask,
                    iou_threshold=0.7
                )
            
            logger.info(f"[Text Outside] Final boxes after merge & filter: {len(boxes_from_mask)}")
            
            # VIS 1: CHỈ text masks (bôi đỏ trên cleaned image)
            vis1_masks_only = image.copy()
            
            mask_colored = np.zeros_like(vis1_masks_only)
            mask_colored[:, :] = [255, 0, 0]
            
            mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
            vis1_masks_only = (vis1_masks_only * (1 - mask_3ch * 0.5) + mask_colored * mask_3ch * 0.5).astype(np.uint8)
            
            # VIS 2: Black canvas + CHỈ vẽ text masks từ VIS 1
            vis2_black_canvas = np.zeros((im_h, im_w, 3), dtype=np.uint8)
            
            # Paste CHỈ text masks lên black canvas
            mask_3ch_full = np.stack([mask] * 3, axis=-1) / 255.0
            mask_colored_full = np.zeros((im_h, im_w, 3), dtype=np.uint8)
            mask_colored_full[:, :] = [255, 0, 0]
            
            vis2_black_canvas = (mask_colored_full * mask_3ch_full * 0.5).astype(np.uint8)
            
            # VIS 3: Filter boxes và vẽ green boxes
            vis3_filtered = await self._create_vis3_filtered_boxes(
                vis2_black_canvas.copy(),
                boxes_from_mask,
                image,
                ocr_service
            )
            
            return [], vis1_masks_only, vis2_black_canvas, vis3_filtered
            
        except Exception as e:
            logger.error(f"[Text Outside] Error: {e}")
            blank = image.copy()
            black = np.zeros_like(blank)
            return [], blank, black, black
    
    async def _create_vis3_filtered_boxes(
        self,
        vis2_canvas: np.ndarray,
        boxes: np.ndarray,
        original_image: np.ndarray,
        ocr_service
    ) -> np.ndarray:
        """
        Tạo VIS 3: Filter boxes theo PanelCleaner approach
        
        Filter pipeline:
        1. Filter theo box area (chỉ xử lý boxes nhỏ < 3000px²)
        2. Run OCR trên boxes nhỏ
        3. Discard boxes match blacklist pattern (symbols, numbers only)
        4. Check aspect ratio cho orientation
        """
        if len(boxes) == 0:
            return vis2_canvas
        
        try:
            logger.info(f"[VIS3 Filter] ══════════════════════════════════════════════")
            logger.info(f"[VIS3 Filter] Starting OCR-based filtering on {len(boxes)} boxes")
            
            OCR_MAX_SIZE = 300000
            BOX_MIN_SIZE = 1500
            
            BLACKLIST_PATTERN = r'^[～．ー！？０-９~.!?0-9\-\s]*$'
            
            logger.info(f"[VIS3 Filter] Filter parameters:")
            logger.info(f"[VIS3 Filter]   - BOX_MIN_SIZE={BOX_MIN_SIZE}")
            logger.info(f"[VIS3 Filter]   - OCR_MAX_SIZE={OCR_MAX_SIZE}")
            logger.info(f"[VIS3 Filter]   - BLACKLIST_PATTERN={BLACKLIST_PATTERN}")
            
            filtered_boxes = []
            filtered_texts = []
            filtered_confidences = []
            rejected_boxes = []
            
            filter_stats = {
                'total': len(boxes),
                'too_small': 0,
                'large_accepted': 0,
                'poor_image_quality': 0,
                'ocr_failed': 0,
                'blacklist_match': 0,
                'bad_aspect_ratio': 0,
                'invalid_text': 0,
                'accepted': 0,
                'avg_confidence': 0.0
            }
            
            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                logger.info(f"[VIS3] Processing Box #{box_idx}: [{x1},{y1},{x2},{y2}] area={area}px²")
                
                if area < BOX_MIN_SIZE:
                    filter_stats['too_small'] += 1
                    rejected_boxes.append((box, box_idx, 'too_small'))  # THÊM
                    logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} too small (area={area} < {BOX_MIN_SIZE})")
                    continue
                
                if area >= OCR_MAX_SIZE:
                    aspect_ratio = width / max(height, 1)
                    if 0.05 <= aspect_ratio <= 30.0:
                        filtered_boxes.append((box, box_idx))
                        filtered_texts.append("[LARGE BOX - SKIPPED OCR]")
                        filtered_confidences.append({
                            'confidence': 50.0,
                            'quality': 'fair',
                            'metrics': {'text_length': 0, 'char_density': 0.0, 'has_meaningful_chars': False, 'language_detected': 'skipped', 'special_char_ratio': 0.0}
                        })
                        filter_stats['large_accepted'] += 1
                        logger.debug(f"[VIS3 Filter] ✓ Box #{box_idx} Large box accepted (area={area}, AR={aspect_ratio:.2f})")
                    else:
                        filter_stats['bad_aspect_ratio'] += 1
                        rejected_boxes.append((box, box_idx, 'bad_aspect_ratio'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Large box rejected (bad AR={aspect_ratio:.2f})")
                    continue
                
                if ocr_service.ocr_model is None:
                    aspect_ratio = width / max(height, 1)
                    if 0.1 <= aspect_ratio <= 20.0:
                        filtered_boxes.append((box, box_idx))
                        filtered_texts.append("[NO OCR MODEL]")
                        filtered_confidences.append({
                            'confidence': 30.0,
                            'quality': 'poor',
                            'metrics': {'text_length': 0, 'char_density': 0.0, 'has_meaningful_chars': False, 'language_detected': 'unknown', 'special_char_ratio': 0.0}
                        })
                        filter_stats['accepted'] += 1
                        logger.debug(f"[VIS3 Filter] ⚠ Box #{box_idx} No OCR model, accepting box (area={area}, AR={aspect_ratio:.2f})")
                    else:
                        filter_stats['bad_aspect_ratio'] += 1
                        rejected_boxes.append((box, box_idx, 'bad_aspect_ratio'))  # THÊM
                    continue
                
                # Crop region với padding
                padding = 2
                x1_crop = max(0, x1 - padding)
                y1_crop = max(0, y1 - padding)
                x2_crop = min(original_image.shape[1], x2 + padding)
                y2_crop = min(original_image.shape[0], y2 + padding)
                
                if x2_crop <= x1_crop or y2_crop <= y1_crop:
                    continue
                
                cropped = original_image[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if cropped.size == 0:
                    continue
                
                # DEBUG: Save cropped image for inspection
                try:
                    from app.utils.image_utils import save_temp_image
                    debug_path = save_temp_image(cropped, f"debug_box{box_idx}")
                    logger.debug(f"[VIS3 Debug] Box #{box_idx} cropped saved: /temp/{os.path.basename(debug_path)}")
                except:
                    pass
                
                # BƯỚC MỚI: Check image quality TRƯỚC KHI chạy OCR
                quality_metrics = self._calculate_image_quality(cropped)
                
                logger.debug(f"[VIS3 Quality] Box #{box_idx} Image Quality: blur={quality_metrics['variance']:.1f}, brightness={quality_metrics['mean_brightness']:.1f}, contrast={quality_metrics['contrast']:.1f}, sharpness={quality_metrics['sharpness_score']:.1f}%")
                
                if not quality_metrics['is_acceptable']:
                    filter_stats['invalid_text'] += 1
                    logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Poor image quality (area={area}) | variance={quality_metrics['variance']:.1f} < 100, contrast={quality_metrics['contrast']:.1f}")
                    continue
                
                try:
                    text = await ocr_service._run_ocr(cropped)
                    
                    logger.info(f"[VIS3 OCR] Box #{box_idx} [{x1},{y1},{x2},{y2}] area={area}px² → OCR: '{text[:100] if text else 'NONE'}'")
                    
                    aspect_ratio = width / max(height, 1)
                    if aspect_ratio < 0.03 or aspect_ratio > 30.0:
                        filter_stats['bad_aspect_ratio'] += 1
                        rejected_boxes.append((box, box_idx, 'bad_aspect_ratio'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Bad AR={aspect_ratio:.2f} (area={area}) | OCR: '{text[:50] if text else 'NONE'}'")
                        continue
                    
                    if not text or text == "[OCR ERROR]":
                        filter_stats['ocr_failed'] += 1
                        rejected_boxes.append((box, box_idx, 'ocr_failed'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} OCR failed (area={area}) | Result: '{text}'")
                        continue
                    
                    # THÊM: Check minimum characters cho large boxes
                    if area > 20000:
                        min_chars_for_large_box = 10
                        if len(text.strip()) < min_chars_for_large_box:
                            filter_stats['invalid_text'] += 1
                            rejected_boxes.append((box, box_idx, 'too_few_chars'))  # THÊM
                            logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Large box with too few chars: {len(text.strip())} < {min_chars_for_large_box} (area={area}) | OCR: '{text[:50]}'")
                            continue
                    
                    text_stripped = text.strip()
                    
                    if len(text_stripped) < 5:
                        filter_stats['invalid_text'] += 1
                        rejected_boxes.append((box, box_idx, 'text_too_short'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Text too short: '{text_stripped}' (area={area}) | len={len(text_stripped)}")
                        continue
                    
                    # BƯỚC 3: Check text validity
                    is_valid = ocr_service._is_valid_text(text_stripped)
                    
                    if not is_valid:
                        filter_stats['invalid_text'] += 1
                        rejected_boxes.append((box, box_idx, 'invalid_text'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Invalid text: '{text_stripped}' (area={area}) | Failed validity check")
                        continue
                    
                    # BƯỚC 4: Check text density (chars per pixel²)
                    text_density = self._calculate_text_density(text_stripped, area)
                    min_density = 0.0002
                    
                    if text_density < min_density:
                        filter_stats['invalid_text'] += 1
                        rejected_boxes.append((box, box_idx, 'low_density'))  # THÊM
                        logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} Low text density: {text_density:.6f} < {min_density} (area={area}, text_len={len(text_stripped)}) | OCR: '{text_stripped[:30]}'")
                        continue
                    
                    # Calculate OCR confidence
                    ocr_quality = ocr_service.calculate_ocr_confidence(text_stripped, area)
                    confidence_score = ocr_quality['confidence']
                    quality_level = ocr_quality['quality']
                    
                    filtered_boxes.append((box, box_idx))
                    filtered_texts.append(text_stripped)
                    filtered_confidences.append(ocr_quality)
                    filter_stats['accepted'] += 1
                    
                    logger.info(f"[VIS3 Filter] ✓ ACCEPTED | Box #{box_idx} [{x1},{y1},{x2},{y2}] area={area}px² AR={aspect_ratio:.2f}")
                    logger.info(f"[VIS3 Quality] Confidence: {confidence_score:.1f}% ({quality_level}) | Text: '{text_stripped[:50]}'")
                    logger.info(f"[VIS3 Quality] Metrics: len={ocr_quality['metrics']['text_length']}, density={ocr_quality['metrics']['char_density']:.6f}, lang={ocr_quality['metrics']['language_detected']}")
                    
                except Exception as e:
                    filter_stats['ocr_failed'] += 1
                    rejected_boxes.append((box, box_idx, 'ocr_exception'))  # THÊM
                    logger.debug(f"[VIS3 Filter] ✗ Box #{box_idx} OCR Exception (area={area}) | Error: {str(e)[:50]}")
                    continue
            
            # Calculate average confidence
            if len(filtered_confidences) > 0:
                avg_conf = sum(c['confidence'] for c in filtered_confidences) / len(filtered_confidences)
                filter_stats['avg_confidence'] = avg_conf
            
            logger.info(f"[VIS3 Filter] ──────────────────────────────────────────")
            logger.info(f"[VIS3 Filter] Filter summary:")
            logger.info(f"[VIS3 Filter]   Total input boxes: {filter_stats['total']}")
            logger.info(f"[VIS3 Filter]   ✗ Too small (< {BOX_MIN_SIZE}px²): {filter_stats['too_small']}")
            logger.info(f"[VIS3 Filter]   ✓ Large boxes accepted (> {OCR_MAX_SIZE}px²): {filter_stats['large_accepted']}")
            logger.info(f"[VIS3 Filter]   ✗ Poor image quality (blur/low contrast): {filter_stats['poor_image_quality']}")
            logger.info(f"[VIS3 Filter]   ✗ OCR failed: {filter_stats['ocr_failed']}")
            logger.info(f"[VIS3 Filter]   ✗ Blacklist match: {filter_stats['blacklist_match']}")
            logger.info(f"[VIS3 Filter]   ✗ Bad aspect ratio: {filter_stats['bad_aspect_ratio']}")
            logger.info(f"[VIS3 Filter]   ✗ Invalid text: {filter_stats['invalid_text']}")
            logger.info(f"[VIS3 Filter]   ✓ ACCEPTED: {filter_stats['accepted']}")
            logger.info(f"[VIS3 Filter] Final: {len(boxes)} -> {len(filtered_boxes)} boxes")
            
            if len(filtered_confidences) > 0:
                logger.info(f"[VIS3 Filter] Average Confidence: {filter_stats['avg_confidence']:.1f}%")
                quality_counts = {}
                for conf in filtered_confidences:
                    q = conf['quality']
                    quality_counts[q] = quality_counts.get(q, 0) + 1
                logger.info(f"[VIS3 Filter] Quality distribution: {quality_counts}")
            
            if len(filtered_texts) > 0:
                sample_texts = filtered_texts[:5]
                logger.info(f"[VIS3 Filter] Sample accepted texts: {sample_texts}")
            
            vis3_canvas = vis2_canvas.copy()
            
            for idx, (box, box_idx) in enumerate(filtered_boxes):
                x1, y1, x2, y2 = box
                
                # Get confidence for this box
                confidence = filtered_confidences[idx]['confidence']
                quality = filtered_confidences[idx]['quality']
                
                # Color based on quality
                if quality == 'excellent':
                    box_color = (0, 255, 0)  # Green
                elif quality == 'good':
                    box_color = (0, 255, 255)  # Yellow
                elif quality == 'fair':
                    box_color = (0, 165, 255)  # Orange
                else:
                    box_color = (0, 0, 255)  # Red
                
                cv2.rectangle(vis3_canvas, (x1, y1), (x2, y2), box_color, thickness=2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_label = f"#{box_idx} {confidence:.0f}%"
                
                (text_width, text_height), baseline = cv2.getTextSize(text_label, font, font_scale, font_thickness)
                
                text_x = x1 + 5
                text_y = y1 + text_height + 5
                
                # Background for text
                cv2.rectangle(vis3_canvas, (text_x - 2, text_y - text_height - 2), 
                            (text_x + text_width + 2, text_y + baseline + 2), 
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis3_canvas, text_label, (text_x, text_y), 
                           font, font_scale, box_color, font_thickness, cv2.LINE_AA)
            
            # VẼ REJECTED BOXES MÀU VÀNG
            for box, box_idx, reject_reason in rejected_boxes:
                x1, y1, x2, y2 = box
                
                # Màu vàng cho rejected boxes
                yellow_color = (0, 255, 255)  # Yellow in BGR
                
                # Vẽ box màu vàng
                cv2.rectangle(vis3_canvas, (x1, y1), (x2, y2), yellow_color, thickness=2)
                
                # Vẽ label
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_label = f"#{box_idx} X"
                
                (text_width, text_height), baseline = cv2.getTextSize(text_label, font, font_scale, font_thickness)
                
                text_x = x1 + 5
                text_y = y1 + text_height + 5
                
                # Background for text
                cv2.rectangle(vis3_canvas, (text_x - 2, text_y - text_height - 2), 
                            (text_x + text_width + 2, text_y + baseline + 2), 
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis3_canvas, text_label, (text_x, text_y), 
                           font, font_scale, yellow_color, font_thickness, cv2.LINE_AA)
            
            logger.info(f"[VIS3 Filter] ══════════════════════════════════════════════")
            
            return vis3_canvas
            
        except Exception as e:
            logger.error(f"[VIS3 Filter] Error: {e}")
            return vis2_canvas
    
    def _letterbox(self, img, new_shape=(1024, 1024), color=(0, 0, 0), stride=64):
        """Letterbox resize for text detection"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dh, dw = int(dh), int(dw)

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=color)
        
        return img, (r, r), (dw, dh)
    
    def _non_max_suppression_text(self, prediction, conf_thres=0.25, iou_thres=0.35):
        """NMS for text detection"""
        output = [None] * len(prediction)
        
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
            
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                conf_thres,
                iou_thres
            )
            
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                indices = indices.flatten()
                output[i] = pred[indices]
        
        return output
    
    def _postprocess_text_detection(self, outputs, ratio, dw, dh, im_w, im_h):
        """Postprocess text detection results"""
        blks = outputs[0]
        mask = outputs[1]
        lines_map = outputs[2]
        
        # Swap mask và lines_map nếu cần
        if mask.shape[1] == 2:
            mask, lines_map = lines_map, mask
        
        # Apply NMS
        det = self._non_max_suppression_text(blks, 0.4, 0.35)[0]
        
        # Process mask
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        elif len(mask.shape) == 3:
            mask = mask[0]
        
        mask = (mask > 0.3) * 255
        mask = mask.astype(np.uint8)
        
        # Remove padding
        dw_int, dh_int = int(dw), int(dh)
        mask = mask[:mask.shape[0] - dh_int, :mask.shape[1] - dw_int]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        
        # Process boxes nếu có detection
        boxes = []
        scores = []
        if det is not None and len(det) > 0:
            # det format: [x_center, y_center, width, height, score]
            # Convert to corner format [x1, y1, x2, y2]
            det_boxes = det[..., :4].copy()
            det_boxes[..., 0] = det[..., 0] - det[..., 2] / 2  # x1 = x_center - width/2
            det_boxes[..., 1] = det[..., 1] - det[..., 3] / 2  # y1 = y_center - height/2
            det_boxes[..., 2] = det[..., 0] + det[..., 2] / 2  # x2 = x_center + width/2
            det_boxes[..., 3] = det[..., 1] + det[..., 3] / 2  # y2 = y_center + height/2
            
            # Scale to original image size
            resize_ratio = (im_w / (1024 - dw), im_h / (1024 - dh))
            det_boxes[..., [0, 2]] = det_boxes[..., [0, 2]] * resize_ratio[0]
            det_boxes[..., [1, 3]] = det_boxes[..., [1, 3]] * resize_ratio[1]
            
            # Clamp boxes to image boundaries
            det_boxes[..., 0] = np.clip(det_boxes[..., 0], 0, im_w)
            det_boxes[..., 1] = np.clip(det_boxes[..., 1], 0, im_h)
            det_boxes[..., 2] = np.clip(det_boxes[..., 2], 0, im_w)
            det_boxes[..., 3] = np.clip(det_boxes[..., 3], 0, im_h)
            
            boxes = det_boxes.astype(np.int32)
            scores = np.round(det[..., 4], 3)
        
        return boxes, mask, scores
    
    def _extract_boxes_from_mask(
        self, 
        mask: np.ndarray, 
        min_area: int = 100,
        max_area: int = 50000,
        min_aspect_ratio: float = 0.1,
        max_aspect_ratio: float = 20.0,
        min_solidity: float = 0.3,
        merge_kernel_size: int = 3
    ) -> np.ndarray:
        """
        Trích xuất bounding boxes từ mask với filtering hierarchy
        
        Args:
            mask: Binary mask (H, W)
            min_area: Diện tích tối thiểu (default: 100px²)
            max_area: Diện tích tối đa (default: 50000px²)
            min_aspect_ratio: Aspect ratio tối thiểu (default: 0.1)
            max_aspect_ratio: Aspect ratio tối đa (default: 20.0)
            min_solidity: Solidity tối thiểu (default: 0.3)
            merge_kernel_size: Kernel size cho morphology (default: 3)
        
        Returns:
            np.ndarray: Array of boxes [x1, y1, x2, y2]
        """
        try:
            logger.info(f"[Extract Boxes] ──────────────────────────────────────────")
            logger.info(f"[Extract Boxes] Starting box extraction with parameters:")
            logger.info(f"[Extract Boxes]   - min_area={min_area}, max_area={max_area}")
            logger.info(f"[Extract Boxes]   - aspect_ratio=[{min_aspect_ratio}, {max_aspect_ratio}]")
            logger.info(f"[Extract Boxes]   - min_solidity={min_solidity}")
            logger.info(f"[Extract Boxes]   - merge_kernel_size={merge_kernel_size}")
            
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_kernel_size, merge_kernel_size))
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_closed = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask_closed, connectivity=8
            )
            
            logger.info(f"[Extract Boxes] Found {num_labels-1} connected components")
            
            boxes = []
            filter_stats = {
                'total': num_labels - 1,
                'rejected_area': 0,
                'rejected_aspect_ratio': 0,
                'rejected_solidity': 0,
                'accepted': 0
            }
            
            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area < min_area or area > max_area:
                    filter_stats['rejected_area'] += 1
                    continue
                
                aspect_ratio = w / max(h, 1)
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    filter_stats['rejected_aspect_ratio'] += 1
                    continue
                
                bbox_area = w * h
                solidity = area / max(bbox_area, 1)
                if solidity < min_solidity:
                    filter_stats['rejected_solidity'] += 1
                    continue
                
                boxes.append([x, y, x + w, y + h])
                filter_stats['accepted'] += 1
            
            boxes_array = np.array(boxes, dtype=np.int32) if boxes else np.array([])
            
            logger.info(f"[Extract Boxes] Filter results:")
            logger.info(f"[Extract Boxes]   ✓ Accepted: {filter_stats['accepted']}")
            logger.info(f"[Extract Boxes]   ✗ Rejected by area: {filter_stats['rejected_area']}")
            logger.info(f"[Extract Boxes]   ✗ Rejected by aspect ratio: {filter_stats['rejected_aspect_ratio']}")
            logger.info(f"[Extract Boxes]   ✗ Rejected by solidity: {filter_stats['rejected_solidity']}")
            logger.info(f"[Extract Boxes] Final: {num_labels-1} components → {len(boxes_array)} valid boxes")
            
            return boxes_array
            
        except Exception as e:
            logger.error(f"[Extract Boxes] Error: {e}")
            return np.array([])
    
    def _merge_nearby_boxes(
        self,
        boxes: np.ndarray,
        distance_threshold: int = 50,
        direction: str = "both"
    ) -> np.ndarray:
        """
        Merge boxes gần nhau theo khoảng cách
        
        Args:
            boxes: Array of boxes [x1, y1, x2, y2]
            distance_threshold: Khoảng cách tối đa để merge (pixels)
            direction: "vertical", "horizontal", hoặc "both"
        
        Returns:
            np.ndarray: Merged boxes
        """
        if len(boxes) == 0:
            return boxes
        
        try:
            merged = []
            used = set()
            
            for i in range(len(boxes)):
                if i in used:
                    continue
                
                current_box = boxes[i].copy()
                x1, y1, x2, y2 = current_box
                merged_any = True
                
                while merged_any:
                    merged_any = False
                    
                    for j in range(len(boxes)):
                        if j in used or j == i:
                            continue
                        
                        bx1, by1, bx2, by2 = boxes[j]
                        
                        # Calculate distances
                        vertical_distance = min(
                            abs(y1 - by2),  # Current top - Other bottom
                            abs(by1 - y2)   # Other top - Current bottom
                        )
                        
                        horizontal_distance = min(
                            abs(x1 - bx2),  # Current left - Other right
                            abs(bx1 - x2)   # Other left - Current right
                        )
                        
                        # Check overlap
                        x_overlap = max(0, min(x2, bx2) - max(x1, bx1))
                        y_overlap = max(0, min(y2, by2) - max(y1, by1))
                        
                        should_merge = False
                        
                        if direction in ["vertical", "both"]:
                            # Merge nếu gần nhau theo chiều dọc VÀ có overlap ngang
                            if vertical_distance <= distance_threshold and x_overlap > 0:
                                should_merge = True
                        
                        if direction in ["horizontal", "both"]:
                            # Merge nếu gần nhau theo chiều ngang VÀ có overlap dọc
                            if horizontal_distance <= distance_threshold and y_overlap > 0:
                                should_merge = True
                        
                        if should_merge:
                            # Merge boxes
                            x1 = min(x1, bx1)
                            y1 = min(y1, by1)
                            x2 = max(x2, bx2)
                            y2 = max(y2, by2)
                            current_box = [x1, y1, x2, y2]
                            used.add(j)
                            merged_any = True
                
                merged.append(current_box)
                used.add(i)
            
            merged_array = np.array(merged, dtype=np.int32) if merged else np.array([])
            
            logger.info(f"[Merge Boxes] Merged {len(boxes)} -> {len(merged_array)} boxes (threshold={distance_threshold}px)")
            
            return merged_array
            
        except Exception as e:
            logger.error(f"[Merge Boxes] Error: {e}")
            return boxes
    
    def _filter_overlapping_boxes(
        self,
        boxes: np.ndarray,
        iou_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Filter boxes overlap - giữ box lớn hơn, loại box nhỏ bị overlap
        
        Args:
            boxes: Array of boxes [x1, y1, x2, y2]
            iou_threshold: IoU threshold để coi là overlap (default: 0.7)
        
        Returns:
            np.ndarray: Filtered boxes
        """
        if len(boxes) == 0:
            return boxes
        
        try:
            # Sort boxes by area (descending)
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            sorted_indices = np.argsort(areas)[::-1]
            
            keep = []
            removed = []
            
            for i in sorted_indices:
                box_i = boxes[i]
                x1_i, y1_i, x2_i, y2_i = box_i
                area_i = areas[i]
                
                should_keep = True
                
                for kept_box in keep:
                    x1_k, y1_k, x2_k, y2_k = kept_box
                    
                    # Calculate intersection
                    ix1 = max(x1_i, x1_k)
                    iy1 = max(y1_i, y1_k)
                    ix2 = min(x2_i, x2_k)
                    iy2 = min(y2_i, y2_k)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        intersection = (ix2 - ix1) * (iy2 - iy1)
                        
                        # Calculate IoU (intersection over smaller box)
                        iou = intersection / min(area_i, (x2_k - x1_k) * (y2_k - y1_k))
                        
                        if iou > iou_threshold:
                            should_keep = False
                            removed.append(i)
                            break
                
                if should_keep:
                    keep.append(box_i)
            
            filtered = np.array(keep, dtype=np.int32) if keep else np.array([])
            
            logger.info(f"[Filter Overlap] Filtered {len(boxes)} -> {len(filtered)} boxes (removed {len(removed)} overlapping boxes)")
            
            return filtered
            
        except Exception as e:
            logger.error(f"[Filter Overlap] Error: {e}")
            return boxes
    
    def _calculate_text_density(
        self,
        text: str,
        box_area: int
    ) -> float:
        """
        Tính text density (số ký tự / diện tích)
        
        Args:
            text: OCR text
            box_area: Diện tích box (pixels)
        
        Returns:
            float: Text density (chars/px²)
        """
        if not text or box_area == 0:
            return 0.0
        
        # Đếm ký tự có nghĩa (bỏ khoảng trắng, dấu câu đơn)
        meaningful_chars = 0
        for char in text:
            if char.isalpha() or '\u4e00' <= char <= '\u9fff' or \
               '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' or \
               '\uac00' <= char <= '\ud7af':
                meaningful_chars += 1
        
        return meaningful_chars / box_area
    
    def _calculate_image_quality(
        self,
        image: np.ndarray
    ) -> dict:
        """
        Tính toán image quality metrics
        
        Args:
            image: Cropped image region (RGB)
        
        Returns:
            dict: {
                'variance': float,  # Variance of Laplacian (blur detection)
                'mean_brightness': float,  # Average brightness
                'contrast': float,  # Standard deviation of brightness
                'sharpness_score': float,  # 0-100
                'is_acceptable': bool
            }
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # 1. Blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Brightness analysis
            mean_brightness = np.mean(gray)
            
            # 3. Contrast analysis (standard deviation)
            contrast = np.std(gray)
            
            # 4. Calculate sharpness score (0-100)
            # Laplacian variance thresholds:
            # < 50: very blurry
            # 50-100: blurry
            # 100-500: acceptable
            # > 500: sharp
            if laplacian_var >= 500:
                sharpness_score = 100
            elif laplacian_var >= 100:
                sharpness_score = 50 + (laplacian_var - 100) / 400 * 50
            elif laplacian_var >= 50:
                sharpness_score = 25 + (laplacian_var - 50) / 50 * 25
            else:
                sharpness_score = laplacian_var / 50 * 25
            
            # 5. Quality thresholds
            MIN_LAPLACIAN_VAR = 100  # Minimum sharpness
            MIN_CONTRAST = 15        # Minimum contrast
            MIN_BRIGHTNESS = 20      # Too dark
            MAX_BRIGHTNESS = 235     # Too bright (washed out)
            
            is_acceptable = (
                laplacian_var >= MIN_LAPLACIAN_VAR and
                contrast >= MIN_CONTRAST and
                MIN_BRIGHTNESS <= mean_brightness <= MAX_BRIGHTNESS
            )
            
            return {
                'variance': round(laplacian_var, 2),
                'mean_brightness': round(mean_brightness, 2),
                'contrast': round(contrast, 2),
                'sharpness_score': round(sharpness_score, 2),
                'is_acceptable': is_acceptable
            }
            
        except Exception as e:
            logger.error(f"[Image Quality] Error: {e}")
            return {
                'variance': 0.0,
                'mean_brightness': 0.0,
                'contrast': 0.0,
                'sharpness_score': 0.0,
                'is_acceptable': False
            }