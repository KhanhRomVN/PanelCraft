import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Optional
import logging

from app.models.pipeline_models import SegmentData
from app.utils.geometry_utils import find_largest_inscribed_rectangle, apply_nms, filter_segments_by_quality

logger = logging.getLogger(__name__)

class SegmentationService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.session = None
        self._load_model()
    
    def _load_model(self):
        """Load segmentation model"""
        model_path = os.path.join(self.model_base_path, "segmentation", "manga_bubble_seg.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Segmentation model not found: {model_path}")
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("Segmentation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise
    
    async def process(self, image: np.ndarray) -> Tuple[List[SegmentData], Optional[np.ndarray], List[np.ndarray]]:
        """
        Process image through segmentation model
        Returns segments, visualization image, và list masks
        """
        if self.session is None:
            raise RuntimeError("Segmentation model not loaded")
        
        try:
            original_h, original_w = image.shape[:2]
            
            # Preprocess
            input_tensor, scale, pad_w, pad_h = self._preprocess(image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Postprocess
            segments, masks = self._postprocess(outputs, scale, pad_w, pad_h, original_w, original_h, image)
            
            # Filter segments by quality using standard deviation
            if len(segments) > 0:
                filtered_segments, quality_stats = filter_segments_by_quality(
                    segments, 
                    std_threshold=2.0,
                    min_area=100
                )
                
                # Update segments và masks
                filtered_indices = [seg.id for seg in filtered_segments]
                filtered_masks = [masks[i] for i in range(len(masks)) if i in filtered_indices]
                
                segments = filtered_segments
                masks = filtered_masks
            
            return segments, None, masks
            
        except Exception as e:
            logger.error(f"Segmentation processing error: {e}")
            raise
    
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess image for segmentation model"""
        input_size = 640
        orig_h, orig_w = image.shape[:2]
        
        # Calculate scale
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Letterbox with padding
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_w = (input_size - new_w) // 2
        pad_h = (input_size - new_h) // 2
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Normalize and prepare for model
        normalized = canvas.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale, pad_w, pad_h
    
    def _postprocess(self, outputs, scale, pad_w, pad_h, original_w, original_h, original_image) -> List[SegmentData]:
        """Postprocess segmentation outputs"""
        boxes_output = outputs[0][0].T  # (8400, 37)
        masks_output = outputs[1]  # (1, 32, 160, 160)
        
        # Extract components
        boxes = boxes_output[:, :4]
        class_scores = boxes_output[:, 4:5]
        mask_coeffs = boxes_output[:, 5:]
        
        # Filter by confidence
        conf_threshold = 0.5
        valid_mask = (class_scores.squeeze() >= conf_threshold)
        if not np.any(valid_mask):
            return [], []
        
        valid_boxes = boxes[valid_mask]
        valid_scores = class_scores[valid_mask]
        valid_mask_coeffs = mask_coeffs[valid_mask]
        
        # Apply NMS
        nms_indices = apply_nms(valid_boxes, valid_scores.squeeze(), iou_threshold=0.45)
        if len(nms_indices) == 0:
            return [], []
        
        final_boxes = valid_boxes[nms_indices]
        final_scores = valid_scores[nms_indices]
        final_mask_coeffs = valid_mask_coeffs[nms_indices]
        
        # Generate masks and extract segments
        mask_protos = masks_output[0]  # (32, 160, 160)
        mask_protos_reshaped = mask_protos.reshape(32, -1)
        
        segments = []
        masks = []
        
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        for i, (box, score, coeffs) in enumerate(zip(final_boxes, final_scores, final_mask_coeffs)):
            # Generate mask
            mask_logits = np.matmul(coeffs, mask_protos_reshaped)
            mask_logits = mask_logits.reshape(160, 160)
            mask_prob = 1 / (1 + np.exp(-mask_logits))
            
            # Crop mask theo bounding box
            x_center, y_center, width, height = box
            mask_h, mask_w = 160, 160
            x1_mask = int((x_center - width/2) * mask_w / 640)
            y1_mask = int((y_center - height/2) * mask_h / 640)
            x2_mask = int((x_center + width/2) * mask_w / 640)
            y2_mask = int((y_center + height/2) * mask_h / 640)
            
            # Clamp coordinates
            x1_mask, y1_mask = max(0, x1_mask), max(0, y1_mask)
            x2_mask, y2_mask = min(mask_w, x2_mask), min(mask_h, y2_mask)
            
            # Chỉ giữ mask trong bounding box
            mask_cropped = np.zeros((mask_h, mask_w), dtype=np.float32)
            if x2_mask > x1_mask and y2_mask > y1_mask:
                mask_cropped[y1_mask:y2_mask, x1_mask:x2_mask] = mask_prob[y1_mask:y2_mask, x1_mask:x2_mask]
            
            # Resize mask từ 160x160 lên 640x640
            mask_resized_640 = cv2.resize(mask_cropped, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            # Cắt phần padding và resize về kích thước gốc
            mask_no_pad = mask_resized_640[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
            mask_original_size = cv2.resize(mask_no_pad, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            mask_binary = (mask_original_size > 0.3).astype(np.uint8)
            
            # Find contours to get bounding box
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)
            x1, y1, w, h = cv2.boundingRect(largest_contour)
            x2, y2 = x1 + w, y1 + h
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(original_w, x2), min(original_h, y2)
            
            # Crop original image region
            cropped_original = original_image[y1:y2, x1:x2].copy()
            cropped_mask = mask_binary[y1:y2, x1:x2]
            
            if cropped_original.size == 0:
                continue
            
            # KHÔNG tính rectangle ở đây nữa, sẽ tính sau khi có text boxes
            segments.append(SegmentData(
                id=i,
                box=[x1, y1, x2, y2],
                score=float(score),
                rectangles=[]
            ))
            
            # Lưu mask
            masks.append(mask_binary)
        
        return segments, masks
    
    def calculate_rectangles_with_text_boxes(
        self,
        segments: List[SegmentData],
        masks: List[np.ndarray],
        text_boxes_per_segment: List[np.ndarray],
        original_image: np.ndarray
    ) -> List[SegmentData]:
        """
        Tính rectangles cho segments dựa vào text boxes đã detect
        
        Logic mới:
        - Nếu ≥2 text boxes: Tạo 1 rectangle cho MỖI text box
        - Nếu 1 text box: Tạo 1 rectangle từ default algorithm
        - Nếu 0 text boxes: Tạo 1 rectangle từ default algorithm
        
        Args:
            segments: List segments từ segmentation
            masks: List masks tương ứng
            text_boxes_per_segment: List array text boxes cho từng segment
            original_image: Ảnh gốc
        
        Returns:
            List[SegmentData]: Segments đã có rectangles (list)
        """
        logger.info(f"[Rectangle Calculation] ═══════════════════════════════════════")
        logger.info(f"[Rectangle Calculation] Calculating rectangles for {len(segments)} segments")
        
        updated_segments = []
        
        for idx, seg in enumerate(segments):
            x1, y1, x2, y2 = seg.box
            mask = masks[idx] if idx < len(masks) else None
            text_boxes = text_boxes_per_segment[idx] if idx < len(text_boxes_per_segment) else np.array([])
            
            num_boxes = len(text_boxes)
            
            logger.info(f"[Rect Calc] Segment #{seg.id} [{x1},{y1},{x2},{y2}]: {num_boxes} text boxes")
            
            rectangles = []
            
            # CASE 1: Multiple text boxes (≥2) → Tạo 1 rectangle cho MỖI text box
            if num_boxes >= 2:
                logger.info(f"[Rect Calc]   → Multiple boxes detected, creating {num_boxes} separate rectangles")
                
                for tb_idx, tb in enumerate(text_boxes):
                    tb_x1, tb_y1, tb_x2, tb_y2 = tb
                    tb_w = tb_x2 - tb_x1
                    tb_h = tb_y2 - tb_y1
                    
                    logger.info(f"[Rect Calc]     Box #{tb_idx}: [{tb_x1},{tb_y1},{tb_x2},{tb_y2}] size={tb_w}x{tb_h}")
                    
                    # Convert text box to rectangle (global coordinates)
                    # Text box đã ở global coordinates rồi, chỉ cần convert format
                    rect_x = tb_x1
                    rect_y = tb_y1
                    rect_w = tb_w
                    rect_h = tb_h
                    
                    # Clamp to segment bounds
                    rect_x = max(x1, rect_x)
                    rect_y = max(y1, rect_y)
                    rect_w = min(x2 - rect_x, rect_w)
                    rect_h = min(y2 - rect_y, rect_h)
                    
                    if rect_w > 0 and rect_h > 0:
                        rectangles.append([rect_x, rect_y, rect_w, rect_h])
                        logger.info(f"[Rect Calc]     ✓ Rectangle #{tb_idx}: [{rect_x},{rect_y},{rect_w},{rect_h}]")
                
                logger.info(f"[Rect Calc]   ✓ Created {len(rectangles)} rectangles for {num_boxes} text boxes")
            
            # CASE 2: Single box hoặc no box → Dùng thuật toán default
            else:
                logger.info(f"[Rect Calc]   → Using default algorithm (inscribed rectangle)")
                
                if mask is not None:
                    cropped_mask = mask[y1:y2, x1:x2]
                    
                    if cropped_mask.size > 0:
                        local_rect = find_largest_inscribed_rectangle(cropped_mask)
                        
                        if local_rect is not None:
                            rect_x, rect_y, rect_w, rect_h = local_rect
                            global_rect_x = x1 + rect_x
                            global_rect_y = y1 + rect_y
                            rectangles.append([global_rect_x, global_rect_y, rect_w, rect_h])
                            
                            logger.info(f"[Rect Calc]   ✓ Default rectangle: [{global_rect_x},{global_rect_y},{rect_w},{rect_h}]")
            
            # Update segment with rectangles (list)
            updated_seg = SegmentData(
                id=seg.id,
                box=seg.box,
                score=seg.score,
                rectangles=rectangles
            )
            
            updated_segments.append(updated_seg)
        
        logger.info(f"[Rectangle Calculation] Completed ✓")
        logger.info(f"[Rectangle Calculation] ═══════════════════════════════════════")
        
        return updated_segments
    
    def _create_step1_visualization(self, image: np.ndarray, segments: List[SegmentData], masks: List[np.ndarray]) -> np.ndarray:
        """
        Tạo visualization cho Step 1: CHỈ green boundaries
        
        Args:
            image: Ảnh gốc
            segments: List segments
            masks: List masks tương ứng
        
        Returns:
            np.ndarray: Visualization với green boundaries
        """
        vis_image = image.copy()    
        
        for i, segment in enumerate(segments):
            if i < len(masks):
                mask = masks[i]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, (0, 255, 0), thickness=3)
        
        return vis_image
    
    def _create_step2_visualization3(self, image: np.ndarray, segments: List[SegmentData], masks: List[np.ndarray]) -> np.ndarray:
        """
        Tạo visualization cho Step 2 Vis 3: Vẽ TẤT CẢ rectangles màu đỏ lên ảnh gốc
        
        Logic mới:
        - Vẽ TẤT CẢ rectangles trong list (hỗ trợ nhiều rectangles cho 1 segment)
        - Mỗi rectangle có label riêng: S{segment_id}_R{rect_idx}
        
        Args:
            image: Ảnh gốc
            segments: List segments với rectangles (list) đã tính
            masks: List masks (không dùng nhưng giữ để tương thích API)
        
        Returns:
            np.ndarray: Visualization với red rectangles
        """
        vis_image = image.copy()
        
        for segment in segments:
            if len(segment.rectangles) > 0:
                for rect_idx, rectangle in enumerate(segment.rectangles):
                    x, y, w, h = rectangle
                    
                    # Vẽ rectangle màu đỏ (thickness=3)
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
                    
                    # Vẽ label cho mỗi rectangle
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    
                    # Nếu segment có nhiều rectangles, thêm index
                    if len(segment.rectangles) > 1:
                        text_label = f"S{segment.id}_R{rect_idx}"
                    else:
                        text_label = f"S{segment.id}"
                    
                    (text_width, text_height), baseline = cv2.getTextSize(text_label, font, font_scale, font_thickness)
                    
                    text_x = x + 5
                    text_y = y + text_height + 10
                    
                    # Background cho text
                    cv2.rectangle(vis_image, (text_x - 2, text_y - text_height - 2), 
                                (text_x + text_width + 2, text_y + baseline + 2), 
                                (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(vis_image, text_label, (text_x, text_y), 
                               font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)
        
        return vis_image