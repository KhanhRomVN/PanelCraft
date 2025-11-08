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
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Process each image
            for idx, image_path in enumerate(self.image_paths):
                try:                    
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
                        
        except Exception as e:
            self.logger.error(f"Error in segmentation thread: {e}")
            self.error_occurred.emit(f"Segmentation error: {str(e)}")
    
    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Chạy segmentation cho 1 ảnh"""
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess image - LƯU LẠI scale và padding
        input_tensor, scale, pad_w, pad_h = self.preprocess(image)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Postprocess output - truyền scale và padding từ preprocess
        result = self.postprocess(outputs, image, orig_h, orig_w, scale, pad_w, pad_h)
        
        return result
    
    def preprocess(self, image: np.ndarray) -> tuple:
        """Preprocess image cho ONNX model - TRẢ VỀ cả scale và padding"""
        input_size = 640
        orig_h, orig_w = image.shape[:2]
        
        # Tính scale (giữ nguyên tỷ lệ)
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize ảnh
        resized = cv2.resize(image, (new_w, new_h))
        
        # Tạo canvas với padding (letterbox)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        pad_w = (input_size - new_w) // 2
        pad_h = (input_size - new_h) // 2
        canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale, pad_w, pad_h
    
    def postprocess(self, outputs: List[np.ndarray], original_image: np.ndarray, orig_h: int, orig_w: int, scale: float, pad_w: int, pad_h: int) -> np.ndarray:
        """Postprocess ONNX output - Vẽ outline trên ảnh gốc"""
        try:
            # YOLOv8 segmentation output:
            # outputs[0]: detection output (1, 37, 8400) - [x, y, w, h, class_score, mask_coeffs...]
            # outputs[1]: mask prototypes (1, 32, 160, 160)
            
            if len(outputs) < 2:
                self.logger.warning("Insufficient outputs from model")
                # Trả về ảnh gốc nếu không có output
                return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
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
                # Trả về ảnh gốc
                return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            valid_boxes = boxes[valid_mask]               # (N, 4)
            valid_scores = class_scores[valid_mask]       # (N, 1)
            valid_mask_coeffs = mask_coeffs[valid_mask]   # (N, 32)
            
            # Apply NMS (Non-Maximum Suppression)
            nms_indices = self._apply_nms(valid_boxes, valid_scores.squeeze(), iou_threshold=0.45)
            
            if len(nms_indices) == 0:
                # Trả về ảnh gốc
                return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            final_boxes = valid_boxes[nms_indices]         # (M, 4)
            final_mask_coeffs = valid_mask_coeffs[nms_indices]  # (M, 32)
                        
            # ========== GENERATE INSTANCE MASKS ==========
            # Lấy mask prototypes
            mask_protos = masks_output[0]  # (32, 160, 160)
            mask_protos_reshaped = mask_protos.reshape(32, -1)  # (32, 25600)
            
            # Tạo combined mask tổng hợp tất cả instances
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

            # SỬ DỤNG scale và padding đã được truyền từ preprocess
            # Tính lại kích thước ảnh sau resize (trước khi pad)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
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
            
            # ========== VẼ OUTLINE VÀ RECTANGLE TRÊN ẢNH GỐC ==========
            # Chuyển ảnh gốc sang RGB để hiển thị
            result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Tìm contours từ mask
            contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Vẽ contours với màu xanh lá, độ dày 3 pixel
                cv2.drawContours(result_image, contours, -1, (0, 255, 0), thickness=3)
                
                # Optional: Vẽ thêm contour bên trong để tạo hiệu ứng viền đậm hơn
                cv2.drawContours(result_image, contours, -1, (0, 200, 0), thickness=1)
                                
                # ========== VẼ HÌNH CHỮ NHẬT LỚN NHẤT BÊN TRONG MỖI BUBBLE ==========
                for contour in contours:
                    # Tạo mask riêng cho contour này
                    single_mask = np.zeros_like(final_mask)
                    cv2.drawContours(single_mask, [contour], -1, 1, -1)
                    
                    # Tìm hình chữ nhật lớn nhất bên trong
                    rect = self._find_largest_inscribed_rectangle(single_mask)
                    
                    if rect is not None:
                        x, y, w, h = rect
                        # Vẽ hình chữ nhật màu đỏ, độ dày 2 pixel
                        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            else:
                self.logger.warning("No contours found in mask")
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Trả về ảnh gốc nếu có lỗi
            return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
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
    
    def _find_largest_inscribed_rectangle(self, mask: np.ndarray) -> tuple:
        """
        Tìm hình chữ nhật nội tiếp có diện tích lớn nhất trong mask
        Sử dụng Distance Transform + Dynamic Programming
        
        Args:
            mask: Binary mask (np.uint8) - 1 cho vùng object, 0 cho background
        
        Returns:
            tuple: (x, y, width, height) hoặc None nếu không tìm thấy
        """
        if mask.sum() == 0:
            return None
        
        # Tính Distance Transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Tìm bounding box của contour để giới hạn vùng tìm kiếm
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Crop mask và distance transform để tối ưu tốc độ
        mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
        dist_crop = dist_transform[y_min:y_max+1, x_min:x_max+1]
        
        h, w = mask_crop.shape
        
        # Largest Rectangle in Histogram algorithm
        max_area = 0
        best_rect = None
        
        # Tạo height histogram
        height_map = np.zeros((h, w), dtype=np.int32)
        
        for i in range(h):
            for j in range(w):
                if mask_crop[i, j] > 0:
                    if i == 0:
                        height_map[i, j] = 1
                    else:
                        height_map[i, j] = height_map[i-1, j] + 1
                else:
                    height_map[i, j] = 0
        
        # Với mỗi hàng, tìm largest rectangle trong histogram
        for i in range(h):
            histogram = height_map[i, :]
            rect = self._largest_rectangle_in_histogram(histogram, i)
            
            if rect is not None:
                x, y, rw, rh = rect
                area = rw * rh
                
                # Kiểm tra xem rectangle có nằm hoàn toàn trong mask không
                if self._is_rectangle_inside_mask(mask_crop, x, y, rw, rh):
                    if area > max_area:
                        max_area = area
                        # Chuyển tọa độ về ảnh gốc
                        best_rect = (x + x_min, y + y_min, rw, rh)
        
        return best_rect
    
    def _largest_rectangle_in_histogram(self, histogram: np.ndarray, row_index: int) -> tuple:
        """
        Tìm hình chữ nhật lớn nhất trong histogram
        
        Args:
            histogram: Array of heights
            row_index: Index của hàng hiện tại
        
        Returns:
            tuple: (x, y, width, height) hoặc None
        """
        stack = []
        max_area = 0
        best_rect = None
        index = 0
        
        while index < len(histogram):
            if not stack or histogram[index] >= histogram[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                height = histogram[top]
                width = index if not stack else index - stack[-1] - 1
                area = height * width
                
                if area > max_area:
                    max_area = area
                    x = stack[-1] + 1 if stack else 0
                    y = row_index - height + 1
                    best_rect = (x, y, width, height)
        
        while stack:
            top = stack.pop()
            height = histogram[top]
            width = index if not stack else index - stack[-1] - 1
            area = height * width
            
            if area > max_area:
                max_area = area
                x = stack[-1] + 1 if stack else 0
                y = row_index - height + 1
                best_rect = (x, y, width, height)
        
        return best_rect
    
    def _is_rectangle_inside_mask(self, mask: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
        """
        Kiểm tra xem hình chữ nhật có nằm hoàn toàn bên trong mask không
        
        Args:
            mask: Binary mask
            x, y, width, height: Rectangle parameters
        
        Returns:
            bool: True nếu rectangle nằm hoàn toàn trong mask
        """
        if x < 0 or y < 0 or x + width > mask.shape[1] or y + height > mask.shape[0]:
            return False
        
        # Lấy vùng rectangle từ mask
        rect_region = mask[y:y+height, x:x+width]
        
        # Kiểm tra xem tất cả pixel trong rectangle có thuộc mask không
        return np.all(rect_region > 0)
    
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


# ========== TEXT DETECTION PROCESSOR ==========

class TextDetectionThread(QThread):
    """Thread để chạy text detection không block UI - Dựa trên test_comictextdetector.py"""
    
    result_ready = Signal(int, QImage)  # index, result image
    progress_updated = Signal(int, int)  # current, total
    error_occurred = Signal(str)

    def __init__(self, image_paths: list, model_path: str):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        # Các thông số từ test_comictextdetector.py
        self.input_size = 1024
        self.conf_thresh = 0.4
        self.nms_thresh = 0.35
        self.mask_thresh = 0.3

    def run(self):
        try:
            # Load ONNX model với OpenCV DNN
            self.model = cv2.dnn.readNetFromONNX(self.model_path)
            
            for idx, image_path in enumerate(self.image_paths):
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.error(f"Failed to load image: {image_path}")
                        continue
                    
                    # Run text detection
                    result_image = self.detect_text(image)
                    
                    # Convert to QImage
                    qimage = self.numpy_to_qimage(result_image)
                    
                    # Emit result
                    self.result_ready.emit(idx, qimage)
                    self.progress_updated.emit(idx + 1, len(self.image_paths))
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                        
        except Exception as e:
            self.logger.error(f"Error in text detection thread: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Text detection error: {str(e)}")

    def letterbox(self, img, new_shape=(1024, 1024), color=(0, 0, 0), stride=64):
        """Letterbox resize - giống test_comictextdetector.py"""
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

    def detect_text(self, image: np.ndarray) -> np.ndarray:
        """Run comic text detection - Logic từ test_comictextdetector.py"""
        im_h, im_w = image.shape[:2]
        
        # Preprocess với letterbox
        img_in, ratio, (dw, dh) = self.letterbox(image, new_shape=self.input_size, stride=64)
        
        # Convert to blob
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        
        # Run inference
        self.model.setInput(img_in)
        output_names = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(output_names)
        
        # Postprocess
        result_image = self.postprocess(outputs, image, ratio, dw, dh, im_w, im_h)
        
        return result_image

    def non_max_suppression(self, prediction, conf_thres=0.4, iou_thres=0.35):
        """NMS cho YOLO output - từ test_comictextdetector.py"""
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

    def postprocess(self, outputs, original_image, ratio, dw, dh, im_w, im_h):
        """Postprocess kết quả detection - Logic từ test_comictextdetector.py"""
        try:
            blks = outputs[0]
            mask = outputs[1]
            lines_map = outputs[2]
            
            # Swap mask và lines_map nếu cần
            if mask.shape[1] == 2:
                mask, lines_map = lines_map, mask
            
            # Apply NMS
            det = self.non_max_suppression(blks, self.conf_thresh, self.nms_thresh)[0]
            
            # Process mask
            if len(mask.shape) == 4:
                mask = mask[0, 0]
            elif len(mask.shape) == 3:
                mask = mask[0]
            
            mask = (mask > self.mask_thresh) * 255
            mask = mask.astype(np.uint8)
            
            # Remove padding
            dw_int, dh_int = int(dw), int(dh)
            mask = mask[:mask.shape[0] - dh_int, :mask.shape[1] - dw_int]
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            
            # Vẽ kết quả lên ảnh gốc (BGR format)
            result_image = original_image.copy()
            
            # Vẽ mask với màu đỏ (overlay)
            colored_mask = np.zeros_like(result_image)
            colored_mask[mask > 0] = [0, 0, 255]  # Red in BGR
            result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
            
            # Vẽ bounding boxes nếu có detection
            if det is not None and len(det) > 0:
                resize_ratio = (im_w / (self.input_size - dw), im_h / (self.input_size - dh))
                det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
                det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
                
                boxes = det[..., 0:4].astype(np.int32)
                scores = np.round(det[..., 4], 3)
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Vẽ confidence score
                    label = f"{score:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result_image, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
                    cv2.putText(result_image, label, (x1, y1 - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert BGR to RGB for Qt display
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            return result_image
            
        except Exception as e:
            self.logger.error(f"Error in postprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """Convert numpy array to QImage"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.copy()


# ========== OCR PROCESSOR ==========

class OCRThread(QThread):
    """Thread để chạy OCR không block UI"""
    
    result_ready = Signal(int, list)  # index, list of text
    progress_updated = Signal(int, int)  # current, total
    error_occurred = Signal(str)

    def __init__(self, image_paths: list):
        super().__init__()
        self.image_paths = image_paths
        self.logger = logging.getLogger(__name__)
        self.ocr_model = None

    def run(self):
        try:
            # Import và khởi tạo Manga OCR
            from manga_ocr import MangaOcr
            self.ocr_model = MangaOcr()
            
            for idx, image_path in enumerate(self.image_paths):
                try:                    
                    # Run OCR
                    text = self.ocr_model(image_path)
                    
                    # Emit result
                    self.result_ready.emit(idx, [text])  # Wrap in list for consistency
                    self.progress_updated.emit(idx + 1, len(self.image_paths))
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {idx}: {e}")
                    continue
                        
        except Exception as e:
            self.logger.error(f"Error in OCR thread: {e}")
            self.error_occurred.emit(f"OCR error: {str(e)}")


# ========== MAIN PIPELINE PROCESSOR ==========

class SegmentationProcessor(QObject):
    """Service để xử lý pipeline: segmentation -> text detection -> OCR"""
    
    # Signals
    result_ready = Signal(int, QImage)
    text_detection_result_ready = Signal(int, QImage)  # Thêm signal mới
    ocr_result_ready = Signal(int, list)  # Thêm signal mới
    progress_updated = Signal(int, int)
    error_occurred = Signal(str)
    completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.segmentation_thread: SegmentationThread = None
        self.text_detection_thread: TextDetectionThread = None
        self.ocr_thread: OCRThread = None
        self.model_manager = ModelManager()
        
        self.current_index = 0
        self.image_paths = []
        self.text_detection_results = {}  # Lưu kết quả text detection
        self.ocr_results = {}  # Lưu kết quả OCR

    def process_images(self, image_paths: List[str]):
        """Bắt đầu xử lý pipeline: segmentation -> text detection -> OCR"""
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            self.logger.warning("Segmentation already in progress")
            return
        
        # Get model path
        model_path = self.get_model_path()
        if not model_path:
            self.error_occurred.emit("Model not found. Please download models first.")
            return
        
        self.image_paths = image_paths
        self.current_index = 0
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        self.segmentation_thread = SegmentationThread(image_paths, model_path)
        self.segmentation_thread.result_ready.connect(self.on_segmentation_result)
        self.segmentation_thread.progress_updated.connect(self.on_progress_updated)
        self.segmentation_thread.error_occurred.connect(self.on_error)
        self.segmentation_thread.finished.connect(self.on_segmentation_completed)
        self.segmentation_thread.start()

    def on_segmentation_completed(self):
        """Khi segmentation hoàn thành, chạy text detection"""
        # Get text detection model path
        text_detection_model_path = self.get_text_detection_model_path()
        if not text_detection_model_path:
            self.error_occurred.emit("Text detection model not found.")
            return
        
        self.text_detection_thread = TextDetectionThread(self.image_paths, text_detection_model_path)
        self.text_detection_thread.result_ready.connect(self.on_text_detection_result)
        self.text_detection_thread.progress_updated.connect(self.on_progress_updated)
        self.text_detection_thread.error_occurred.connect(self.on_error)
        self.text_detection_thread.finished.connect(self.on_text_detection_completed)
        self.text_detection_thread.start()

    def on_text_detection_completed(self):
        """Khi text detection hoàn thành, chạy OCR"""        
        self.ocr_thread = OCRThread(self.image_paths)
        self.ocr_thread.result_ready.connect(self.on_ocr_result)
        self.ocr_thread.progress_updated.connect(self.on_progress_updated)
        self.ocr_thread.error_occurred.connect(self.on_error)
        self.ocr_thread.finished.connect(self.on_ocr_completed)
        self.ocr_thread.start()

    def on_segmentation_result(self, index: int, result_image: QImage):
        """Handle segmentation result"""
        self.result_ready.emit(index, result_image)

    def on_text_detection_result(self, index: int, result_image: QImage):
        """Xử lý kết quả text detection"""
        self.text_detection_results[index] = result_image
        self.text_detection_result_ready.emit(index, result_image)

    def on_ocr_result(self, index: int, texts: list):
        """Xử lý kết quả OCR"""
        self.ocr_results[index] = texts
        self.ocr_result_ready.emit(index, texts)

    def on_ocr_completed(self):
        """Khi OCR hoàn thành, emit completed"""
        self.completed.emit()

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

    def get_text_detection_model_path(self) -> str:
        """Lấy đường dẫn đến model text detection"""
        base_path = self.model_manager.get_model_path()
        if not base_path:
            return ""
        
        model_file = "comictextdetector.pt.onnx"
        model_path = os.path.join(base_path, model_file)
        
        if not os.path.exists(model_path):
            self.logger.error(f"Text detection model file not found: {model_path}")
            return ""
        
        return model_path

    def on_progress_updated(self, current: int, total: int):
        """Forward progress signal"""
        self.progress_updated.emit(current, total)
    
    def on_error(self, error_msg: str):
        """Forward error signal"""
        self.error_occurred.emit(error_msg)