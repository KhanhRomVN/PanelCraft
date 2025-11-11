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
            # Clone ảnh gốc để không modify input
            final_image = original_image.copy()
            
            # Process TỪNG segment riêng biệt
            for segment in segments:
                x1, y1, x2, y2 = segment['box']
                mask = segment['mask']
                
                # Crop segment region từ ảnh gốc
                cropped_region = original_image[y1:y2, x1:x2].copy()
                cropped_mask = mask[y1:y2, x1:x2]
                
                # Detect và remove text CHỈ trong segment này
                cleaned_region = self._remove_text_from_segment(cropped_region, cropped_mask)
                
                # Paste cleaned region back với mask
                mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
                roi = final_image[y1:y2, x1:x2]
                
                if roi.shape[:2] == cleaned_region.shape[:2]:
                    final_image[y1:y2, x1:x2] = np.where(mask_3ch > 0, cleaned_region, roi)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Text detection processing error: {e}")
            raise
    
    def _remove_text_from_segment(
        self, 
        segment_image: np.ndarray,
        segment_mask: np.ndarray
    ) -> np.ndarray:
        """
        Remove text từ MỘT segment bubble
        
        Args:
            segment_image: Ảnh đã crop của segment (RGB)
            segment_mask: Binary mask của segment
        
        Returns:
            np.ndarray: Segment đã clean text
        """
        im_h, im_w = segment_image.shape[:2]
        
        # Preprocess
        img_in, ratio, (dw, dh) = self._letterbox(segment_image, new_shape=1024, stride=64)
        
        # Convert to blob
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        
        # Run inference
        self.model.setInput(img_in)
        output_names = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(output_names)
        
        # Postprocess để lấy mask
        boxes, mask, scores = self._postprocess_text_detection(
            outputs, ratio, dw, dh, im_w, im_h
        )
        
        # Apply segment mask để CHỈ inpaint trong vùng bubble
        mask = cv2.bitwise_and(mask, mask, mask=segment_mask)
        
        # Inpaint để remove text
        segment_bgr = cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(segment_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        
        return cleaned_rgb
    
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
            resize_ratio = (im_w / (1024 - dw), im_h / (1024 - dh))
            det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
            det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
            
            boxes = det[..., 0:4].astype(np.int32)
            scores = np.round(det[..., 4], 3)
        
        return boxes, mask, scores