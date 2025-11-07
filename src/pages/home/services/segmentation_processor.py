from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QImage
from typing import List
import logging
import os
import numpy as np
import onnxruntime as ort
import cv2
from core.model_manager import ModelManager


class SegmentationThread(QThread):
    """Thread để chạy segmentation không block UI"""
    
    result_ready = Signal(int, QImage)  # index, result image
    progress_updated = Signal(int, int)  # current, total
    error_occurred = Signal(str)
    
    def __init__(self, image_paths: List[str], model_path: str):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.session = None
    
    def run(self):
        """Chạy segmentation cho tất cả ảnh"""
        try:
            # Load ONNX model
            self.logger.info(f"Loading ONNX model from: {self.model_path}")
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Process each image
            for idx, image_path in enumerate(self.image_paths):
                try:
                    self.logger.info(f"Processing image {idx + 1}/{len(self.image_paths)}: {image_path}")
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.error(f"Failed to load image: {image_path}")
                        continue
                    
                    # Run segmentation
                    result_image = self.segment_image(image)
                    
                    # Convert to QImage
                    qimage = self.numpy_to_qimage(result_image)
                    
                    # Emit result
                    self.result_ready.emit(idx, qimage)
                    self.progress_updated.emit(idx + 1, len(self.image_paths))
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {idx}: {e}")
                    continue
            
            self.logger.info("Segmentation completed for all images")
            
        except Exception as e:
            self.logger.error(f"Error in segmentation thread: {e}")
            self.error_occurred.emit(f"Segmentation error: {str(e)}")
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Chạy segmentation cho 1 ảnh"""
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Postprocess output
        result = self.postprocess(outputs, orig_h, orig_w)
        
        return result
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image cho ONNX model"""
        # Resize to model input size (typically 640x640 for YOLOv8)
        input_size = 640
        resized = cv2.resize(image, (input_size, input_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess(self, outputs: List[np.ndarray], orig_h: int, orig_w: int) -> np.ndarray:
        """Postprocess ONNX output thành segmentation mask với colored overlay - Instance Segmentation đúng chuẩn YOLOv8"""
        try:
            # YOLOv8 segmentation output:
            # outputs[0]: detection output (1, 37, 8400) - [x, y, w, h, class_score, mask_coeffs...]
            # outputs[1]: mask prototypes (1, 32, 160, 160)
            
            if len(outputs) < 2:
                self.logger.warning("Insufficient outputs from model")
                return np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            
            boxes_output = outputs[0]  # (1, 37, 8400)
            masks_output = outputs[1]  # (1, 32, 160, 160)
            
            # Transpose boxes_output để xử lý
            boxes_output = boxes_output[0]  # (37, 8400)
            boxes_output = boxes_output.T   # (8400, 37)
            
            # Extract components: [x, y, w, h] + [class_score] + [32 mask_coeffs]
            boxes = boxes_output[:, :4]                    # (8400, 4) - bbox
            class_scores = boxes_output[:, 4:5]            # (8400, 1) - confidence
            mask_coeffs = boxes_output[:, 5:]             # (8400, 32) - mask coefficients
            
            # Filter theo confidence threshold
            conf_threshold = 0.5
            valid_mask = (class_scores.squeeze() >= conf_threshold)
            
            if not np.any(valid_mask):
                self.logger.info("No detections above confidence threshold")
                return np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            
            valid_boxes = boxes[valid_mask]               # (N, 4)
            valid_scores = class_scores[valid_mask]       # (N, 1)
            valid_mask_coeffs = mask_coeffs[valid_mask]   # (N, 32)
            
            # Apply NMS (Non-Maximum Suppression)
            nms_indices = self._apply_nms(valid_boxes, valid_scores.squeeze(), iou_threshold=0.45)
            
            if len(nms_indices) == 0:
                self.logger.info("No detections after NMS")
                return np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            
            final_boxes = valid_boxes[nms_indices]         # (M, 4)
            final_mask_coeffs = valid_mask_coeffs[nms_indices]  # (M, 32)
            
            self.logger.info(f"Found {len(final_boxes)} instances after NMS")
            
            # ========== GENERATE INSTANCE MASKS ==========
            # Lấy mask prototypes
            mask_protos = masks_output[0]  # (32, 160, 160)
            mask_protos_reshaped = mask_protos.reshape(32, -1)  # (32, 25600)
            
            # Tạo combined mask tổng hợp tất cả instances
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
            
            # Tính scale và padding từ preprocess
            # Input size = 640x640
            input_size = 640
            scale = min(input_size / orig_w, input_size / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_w = (input_size - new_w) // 2
            pad_h = (input_size - new_h) // 2
            
            for box, coeffs in zip(final_boxes, final_mask_coeffs):
                # Tính mask logits: mask_coeffs @ mask_protos
                mask_logits = np.matmul(coeffs, mask_protos_reshaped)  # (25600,)
                mask_logits = mask_logits.reshape(160, 160)            # (160, 160)
                
                # Áp dụng sigmoid
                mask_prob = 1 / (1 + np.exp(-mask_logits))
                
                # Crop mask theo bounding box (quan trọng!)
                # Box format: [x_center, y_center, width, height] - normalized to 640
                x_center, y_center, width, height = box
                
                # Convert về tọa độ trong mask 160x160
                mask_h, mask_w = 160, 160
                x1 = int((x_center - width/2) * mask_w / 640)
                y1 = int((y_center - height/2) * mask_h / 640)
                x2 = int((x_center + width/2) * mask_w / 640)
                y2 = int((y_center + height/2) * mask_h / 640)
                
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(mask_w, x2), min(mask_h, y2)
                
                # Chỉ giữ mask trong bounding box
                mask_cropped = np.zeros((mask_h, mask_w), dtype=np.float32)
                if x2 > x1 and y2 > y1:
                    mask_cropped[y1:y2, x1:x2] = mask_prob[y1:y2, x1:x2]
                
                # Resize mask từ 160x160 lên 640x640
                mask_resized_640 = cv2.resize(mask_cropped, (640, 640), interpolation=cv2.INTER_LINEAR)
                
                # Cắt phần padding và resize về kích thước gốc
                mask_no_pad = mask_resized_640[pad_h:pad_h+new_h, pad_w:pad_w+new_w]
                mask_original_size = cv2.resize(mask_no_pad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
                # Threshold và thêm vào combined mask
                mask_binary = (mask_original_size > 0.3).astype(np.float32)
                combined_mask = np.maximum(combined_mask, mask_binary)
            
            # Chuyển sang binary mask
            final_mask = (combined_mask > 0.5).astype(np.uint8)
            
            # Tạo colored overlay (màu xanh lá cho speech balloons)
            colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            colored_mask[final_mask > 0] = [0, 255, 0]  # Green color (RGB)
            
            return colored_mask
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    
    def _apply_nms(self, boxes, scores, iou_threshold=0.45):
        """
        Non-Maximum Suppression để loại bỏ duplicate detections
        
        Args:
            boxes: (N, 4) - [x_center, y_center, width, height] - normalized to 640
            scores: (N,) - confidence scores
            iou_threshold: IoU threshold for NMS
        
        Returns:
            list: Indices của boxes được giữ lại
        """
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes_corner = np.stack([x1, y1, x2, y2], axis=1)
        
        # Sort by scores (descending)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(boxes_corner[i, 0], boxes_corner[order[1:], 0])
            yy1 = np.maximum(boxes_corner[i, 1], boxes_corner[order[1:], 1])
            xx2 = np.minimum(boxes_corner[i, 2], boxes_corner[order[1:], 2])
            yy2 = np.minimum(boxes_corner[i, 3], boxes_corner[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            area_i = (boxes_corner[i, 2] - boxes_corner[i, 0]) * (boxes_corner[i, 3] - boxes_corner[i, 1])
            area_order = (boxes_corner[order[1:], 2] - boxes_corner[order[1:], 0]) * \
                         (boxes_corner[order[1:], 3] - boxes_corner[order[1:], 1])
            
            union = area_i + area_order - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU < threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """Convert numpy array to QImage"""
        height, width = image.shape[:2]
        
        if len(image.shape) == 2:
            # Grayscale
            qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        else:
            # RGB
            bytes_per_line = 3 * width
            qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return qimage.copy()


class SegmentationProcessor(QObject):
    """Service để xử lý segmentation"""
    
    # Signals
    result_ready = Signal(int, QImage)
    progress_updated = Signal(int, int)
    error_occurred = Signal(str)
    completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.segmentation_thread: SegmentationThread = None
        self.model_manager = ModelManager()
    
    def process_images(self, image_paths: List[str]):
        """Bắt đầu xử lý segmentation"""
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            self.logger.warning("Segmentation already in progress")
            return
        
        # Get model path
        model_path = self.get_model_path()
        if not model_path:
            self.error_occurred.emit("Model not found. Please download models first.")
            return
        
        self.logger.info(f"Starting segmentation for {len(image_paths)} images")
        
        self.segmentation_thread = SegmentationThread(image_paths, model_path)
        self.segmentation_thread.result_ready.connect(self.on_result_ready)
        self.segmentation_thread.progress_updated.connect(self.on_progress_updated)
        self.segmentation_thread.error_occurred.connect(self.on_error)
        self.segmentation_thread.finished.connect(self.on_completed)
        self.segmentation_thread.start()
    
    def get_model_path(self) -> str:
        """Lấy đường dẫn đến model ONNX"""
        base_path = self.model_manager.get_model_path()
        if not base_path:
            return ""
        
        model_file = "yolov8_converted.onnx"
        model_path = os.path.join(base_path, model_file)
        
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return ""
        
        return model_path
    
    def on_result_ready(self, index: int, result_image: QImage):
        """Forward result signal"""
        self.result_ready.emit(index, result_image)
    
    def on_progress_updated(self, current: int, total: int):
        """Forward progress signal"""
        self.progress_updated.emit(current, total)
    
    def on_error(self, error_msg: str):
        """Forward error signal"""
        self.error_occurred.emit(error_msg)
    
    def on_completed(self):
        """Handle completion"""
        self.logger.info("Segmentation processing completed")
        self.completed.emit()