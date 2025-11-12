# app/services/text_detection_service.py
import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

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
    ) -> np.ndarray:
        """
        Process segments để remove text và trả về ảnh gốc đã clean
        
        Args:
            original_image: Ảnh gốc (RGB)
            segments: List segments từ segmentation (dạng dict với keys: id, box, mask)
        
        Returns:
            np.ndarray: Ảnh gốc đã clean text trong các bubble
        """
        if self.model is None:
            raise RuntimeError("Text detection model not loaded")
        
        try:
            logger.info(f"[TEXT_DETECTION] Starting text removal for {len(segments)} segments")
            
            # BƯỚC 1: TẠO BLANK CANVAS VỚI CHỈ CÓ BALLOON REGIONS
            original_h, original_w = original_image.shape[:2]
            blank_canvas = np.full((original_h, original_w, 3), 0, dtype=np.uint8)
            
            logger.info(f"[TEXT_DETECTION] Created blank canvas: {blank_canvas.shape}")
            
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
            
            logger.info(f"[TEXT_DETECTION] Blank canvas created with {len(segments)} balloon regions")
            
            # BƯỚC 2: CHẠY TEXT DETECTION TRÊN BLANK CANVAS
            cleaned_canvas, text_mask = self._remove_text_from_canvas(blank_canvas)
            
            logger.info(f"[TEXT_DETECTION] Text detection completed on blank canvas")
            
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
            
            logger.info(f"[TEXT_DETECTION] Completed text removal - pasted back to original")
            return final_image
            
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
            tuple: (cleaned_canvas, text_mask)
        """
        im_h, im_w = canvas.shape[:2]
        
        logger.info(f"[TEXT_DETECTION][Canvas] Input canvas shape: {canvas.shape}")
        
        # Preprocess
        img_in, ratio, (dw, dh) = self._letterbox(canvas, new_shape=1024, stride=64)
        
        logger.info(f"[TEXT_DETECTION][Canvas] After letterbox: {img_in.shape}, ratio: {ratio}, padding: ({dw},{dh})")
        
        # Convert to blob
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        
        # Run inference
        logger.info(f"[TEXT_DETECTION][Canvas] Running comictextdetector.pt.onnx inference on full canvas...")
        self.model.setInput(img_in)
        output_names = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(output_names)
        
        logger.info(f"[TEXT_DETECTION][Canvas] Inference completed. Output count: {len(outputs)}")
        
        # Postprocess
        boxes, mask, scores = self._postprocess_text_detection(
            outputs, ratio, dw, dh, im_w, im_h
        )
        
        logger.info(f"[TEXT_DETECTION][Canvas] Text mask non-zero pixels: {np.count_nonzero(mask)}")
        
        # Inpaint toàn bộ canvas
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(canvas_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        
        logger.info(f"[TEXT_DETECTION][Canvas] Inpainting completed. Output shape: {cleaned_rgb.shape}")
        
        return cleaned_rgb, mask
    
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
    
    def _non_max_suppression_text(self, prediction, conf_thres=0.4, iou_thres=0.35):
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
            logger.info(f"[TEXT_DETECTION][Postprocess] Detected {len(det)} text boxes before coordinate transformation")
            resize_ratio = (im_w / (1024 - dw), im_h / (1024 - dh))
            det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
            det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
            
            boxes = det[..., 0:4].astype(np.int32)
            scores = np.round(det[..., 4], 3)
            logger.info(f"[TEXT_DETECTION][Postprocess] Boxes shape: {boxes.shape}, Scores: {scores}")
        else:
            logger.info(f"[TEXT_DETECTION][Postprocess] No text boxes detected")
        
        return boxes, mask, scores