from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QImage
from typing import List, Dict, Tuple, Optional
import logging
import os
import numpy as np
import cv2
import onnxruntime as ort
from core.model_manager import ModelManager


class MangaPipelineThread(QThread):
    """Thread chạy full pipeline: Segmentation -> Text Detection -> OCR -> Final Composition"""
    
    result_ready = Signal(int, QImage)  # index, final result image (step 5)
    ocr_result_ready = Signal(int, list)  # index, list of OCR texts
    progress_updated = Signal(int, int, str)  # current, total, step_name
    error_occurred = Signal(str)
    
    def __init__(self, image_paths: List[str], segmentation_model_path: str, text_detection_model_path: str, ocr_model_path: str):
        super().__init__()
        self.image_paths = image_paths
        self.segmentation_model_path = segmentation_model_path
        self.text_detection_model_path = text_detection_model_path
        self.ocr_model_path = ocr_model_path
        self.logger = logging.getLogger(__name__)
        
        # Models
        self.segmentation_session = None
        self.text_detection_model = None
        self.ocr_model = None
        self.ocr_processor = None
        self.ocr_tokenizer = None
        
        # Segmentation config
        self.seg_input_size = 640
        self.seg_conf_thresh = 0.5
        self.seg_iou_thresh = 0.45
        
        # Text detection config
        self.text_input_size = 1024
        self.text_conf_thresh = 0.4
        self.text_nms_thresh = 0.35
        self.text_mask_thresh = 0.3
    
    def run(self):
        """Chạy full pipeline cho tất cả ảnh"""
        try:
            # Load segmentation model (ONNX Runtime)
            self.segmentation_session = ort.InferenceSession(
                self.segmentation_model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Load text detection model (OpenCV DNN)
            self.text_detection_model = cv2.dnn.readNetFromONNX(self.text_detection_model_path)
            
            # Load OCR model (PyTorch)
            try:
                from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
                import subprocess
                import sys
                
                # Check và install dependencies nếu cần
                try:
                    import fugashi
                    import unidic_lite
                except ImportError:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fugashi", "unidic-lite"])
                
                # Validate OCR model files
                required_files = ['config.json', 'preprocessor_config.json', 'pytorch_model.bin', 
                                 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt']
                missing_files = []
                
                for file in required_files:
                    file_path = os.path.join(self.ocr_model_path, file)
                    if not os.path.exists(file_path):
                        missing_files.append(file)
                        self.logger.error(f"Missing OCR file: {file}")
                
                if missing_files:
                    raise FileNotFoundError(f"Missing required OCR files: {missing_files}")
                                
                # Load OCR components
                self.ocr_processor = ViTImageProcessor.from_pretrained(self.ocr_model_path)
                self.ocr_tokenizer = AutoTokenizer.from_pretrained(self.ocr_model_path)
                self.ocr_model = VisionEncoderDecoderModel.from_pretrained(self.ocr_model_path)
                self.ocr_model.eval()
                
            except Exception as e:
                self.logger.error(f"Failed to load OCR model: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warning("Continuing without OCR functionality")
                self.ocr_model = None
            
            # Process each image
            total = len(self.image_paths)
            for idx, image_path in enumerate(self.image_paths):
                try:
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.error(f"Failed to load image: {image_path}")
                        continue
                    
                    # Run full pipeline
                    final_result = self.process_single_image(image, idx, total)
                    
                    # Convert to QImage
                    qimage = self.numpy_to_qimage(final_result)
                    
                    # Emit result
                    self.result_ready.emit(idx, qimage)
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                            
        except Exception as e:
            self.logger.error(f"Error in pipeline thread: {e}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(f"Pipeline error: {str(e)}")
    
    def process_single_image(self, image: np.ndarray, idx: int, total: int) -> np.ndarray:
        """
        Xử lý 1 ảnh qua full pipeline (5 steps)
        Returns: Final result image (step 5)
        """
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ========== STEP 1: SEGMENTATION - Extract Balloon Segments ==========
        self.progress_updated.emit(idx + 1, total, "Step 1: Segmentation")
        
        segments = self.extract_balloon_segments(image_rgb)
        
        if not segments:
            self.logger.warning("No segments found, returning original image")
            return image_rgb
                
        # ========== STEP 2: Create Blank Canvas ==========
        self.progress_updated.emit(idx + 1, total, "Step 2: Creating canvas")
        blank_canvas = self.create_blank_canvas_with_balloons(image_rgb, segments)
        
        # ========== STEP 3: TEXT DETECTION & REMOVAL ==========
        self.progress_updated.emit(idx + 1, total, "Step 3: Text detection")
        cleaned_canvas, text_mask = self.remove_text_from_canvas(blank_canvas)
        
        # ========== STEP 4: Update Segments with Cleaned Images ==========
        self.progress_updated.emit(idx + 1, total, "Step 4: Updating segments")
        
        for segment in segments:
            x1, y1, x2, y2 = segment['box']
            segment['cleaned_image'] = cleaned_canvas[y1:y2, x1:x2].copy()
        
        # ========== STEP 5: FINAL COMPOSITION - Paste Back to Original ==========
        self.progress_updated.emit(idx + 1, total, "Step 5: Final composition")        
        final_image = self.paste_cleaned_segments_to_original(image_rgb, segments)
        
        # ========== STEP 6: OCR Processing ==========
        if self.ocr_model is not None:
            self.progress_updated.emit(idx + 1, total, "Step 6: OCR Processing")
            
            ocr_texts = []
            for i, segment in enumerate(segments):
                try:
                    text = self.run_ocr_on_segment(segment['cropped_original'])
                    ocr_texts.append(text)
                except Exception as e:
                    self.logger.error(f"OCR failed for segment {i}: {e}")
                    ocr_texts.append("[OCR ERROR]")
            
            # Emit OCR results
            self.ocr_result_ready.emit(idx, ocr_texts)
        else:
            self.logger.warning("OCR model not loaded - skipping OCR step")
                
        return final_image
    
    # ========== SEGMENTATION METHODS ==========
    
    def extract_balloon_segments(self, image_rgb: np.ndarray) -> List[Dict]:
        """Extract balloon segments using YOLOv8 segmentation"""
        original_h, original_w = image_rgb.shape[:2]
        
        # Preprocess
        input_tensor, scale, pad_w, pad_h = self.preprocess_for_segmentation(image_rgb)
        
        # Inference
        input_name = self.segmentation_session.get_inputs()[0].name
        outputs = self.segmentation_session.run(None, {input_name: input_tensor})
        
        # Postprocess
        segments = self.postprocess_segmentation(outputs, scale, pad_w, pad_h, original_w, original_h, image_rgb)
        
        return segments
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> Tuple:
        """Preprocess image for segmentation model"""
        input_size = self.seg_input_size
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
    
    def postprocess_segmentation(self, outputs, scale, pad_w, pad_h, original_w, original_h, original_image):
        """Postprocess segmentation outputs to extract segments"""
        boxes_output = outputs[0][0].T  # (8400, 37)
        masks_output = outputs[1]  # (1, 32, 160, 160)
        
        # Extract components
        boxes = boxes_output[:, :4]
        class_scores = boxes_output[:, 4:5]
        mask_coeffs = boxes_output[:, 5:]
        
        # Filter by confidence
        valid_mask = (class_scores.squeeze() >= self.seg_conf_thresh)
        if not np.any(valid_mask):
            return []
        
        valid_boxes = boxes[valid_mask]
        valid_scores = class_scores[valid_mask]
        valid_mask_coeffs = mask_coeffs[valid_mask]
        
        # Apply NMS
        nms_indices = self.apply_nms_segmentation(valid_boxes, valid_scores.squeeze(), self.seg_iou_thresh)
        if len(nms_indices) == 0:
            return []
        
        final_boxes = valid_boxes[nms_indices]
        final_scores = valid_scores[nms_indices]
        final_mask_coeffs = valid_mask_coeffs[nms_indices]
        
        # Generate masks
        mask_protos = masks_output[0]  # (32, 160, 160)
        mask_protos_reshaped = mask_protos.reshape(32, -1)
        
        segments = []
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        for i, (box, score, coeffs) in enumerate(zip(final_boxes, final_scores, final_mask_coeffs)):
            # Generate mask
            mask_logits = np.matmul(coeffs, mask_protos_reshaped)
            mask_logits = mask_logits.reshape(160, 160)
            mask_prob = 1 / (1 + np.exp(-mask_logits))
            
            # Crop mask to bounding box
            x_center, y_center, width, height = box
            mask_h, mask_w = 160, 160
            x1 = int((x_center - width/2) * mask_w / 640)
            y1 = int((y_center - height/2) * mask_h / 640)
            x2 = int((x_center + width/2) * mask_w / 640)
            y2 = int((y_center + height/2) * mask_h / 640)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask_w, x2), min(mask_h, y2)
            
            mask_cropped = np.zeros((mask_h, mask_w), dtype=np.float32)
            if x2 > x1 and y2 > y1:
                mask_cropped[y1:y2, x1:x2] = mask_prob[y1:y2, x1:x2]
            
            # Resize mask to original size
            mask_resized_640 = cv2.resize(mask_cropped, (640, 640), interpolation=cv2.INTER_LINEAR)
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
            
            segments.append({
                'id': i,
                'box': [x1, y1, x2, y2],
                'score': float(score),
                'mask': mask_binary,
                'cropped_original': cropped_original,
                'cropped_mask': cropped_mask
            })
        
        return segments
    
    def apply_nms_segmentation(self, boxes, scores, iou_threshold):
        """NMS for segmentation"""
        if len(boxes) == 0:
            return []
        
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes_corner = np.stack([x1, y1, x2, y2], axis=1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
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
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    # ========== CANVAS CREATION ==========
    
    def create_blank_canvas_with_balloons(self, original_image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Create blank canvas with only balloon regions"""
        original_h, original_w = original_image.shape[:2]
        blank_canvas = np.full((original_h, original_w, 3), 0, dtype=np.uint8)
        
        for segment in segments:
            x1, y1, x2, y2 = segment['box']
            mask = segment['mask']
            
            cropped_region = segment['cropped_original'].copy()
            cropped_mask = mask[y1:y2, x1:x2]
            
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
        
        return blank_canvas
    
    # ========== TEXT DETECTION & REMOVAL ==========
    
    def remove_text_from_canvas(self, canvas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove text from canvas using Comic Text Detector"""
        im_h, im_w = canvas.shape[:2]
        
        # Preprocess
        img_in, ratio, (dw, dh) = self.letterbox(canvas, new_shape=self.text_input_size, stride=64)
        
        # Convert to blob
        img_in = img_in.transpose((2, 0, 1))[::-1]
        img_in = np.array([np.ascontiguousarray(img_in)]).astype(np.float32) / 255.0
        
        # Run inference
        self.text_detection_model.setInput(img_in)
        output_names = self.text_detection_model.getUnconnectedOutLayersNames()
        outputs = self.text_detection_model.forward(output_names)
        
        # Postprocess
        boxes, mask, scores = self.postprocess_text_detection(outputs, ratio, dw, dh, im_w, im_h)
        
        # Inpaint to remove text
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cleaned = cv2.inpaint(canvas_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
        
        return cleaned_rgb, mask
    
    def letterbox(self, img, new_shape=(1024, 1024), color=(0, 0, 0), stride=64):
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
    
    def non_max_suppression_text(self, prediction, conf_thres=0.4, iou_thres=0.35):
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
    
    def postprocess_text_detection(self, outputs, ratio, dw, dh, im_w, im_h):
        """Postprocess text detection results"""
        blks = outputs[0]
        mask = outputs[1]
        lines_map = outputs[2]
        
        if mask.shape[1] == 2:
            mask, lines_map = lines_map, mask
        
        det = self.non_max_suppression_text(blks, self.text_conf_thresh, self.text_nms_thresh)[0]
        
        # Process mask
        if len(mask.shape) == 4:
            mask = mask[0, 0]
        elif len(mask.shape) == 3:
            mask = mask[0]
        
        mask = (mask > self.text_mask_thresh) * 255
        mask = mask.astype(np.uint8)
        
        # Remove padding
        dw_int, dh_int = int(dw), int(dh)
        mask = mask[:mask.shape[0] - dh_int, :mask.shape[1] - dw_int]
        mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        
        # Process boxes if detection exists
        boxes = []
        scores = []
        if det is not None and len(det) > 0:
            resize_ratio = (im_w / (self.text_input_size - dw), im_h / (self.text_input_size - dh))
            det[..., [0, 2]] = det[..., [0, 2]] * resize_ratio[0]
            det[..., [1, 3]] = det[..., [1, 3]] * resize_ratio[1]
            
            boxes = det[..., 0:4].astype(np.int32)
            scores = np.round(det[..., 4], 3)
        
        return boxes, mask, scores
    
    # ========== FINAL COMPOSITION ==========
    
    def paste_cleaned_segments_to_original(self, original_image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Paste cleaned balloon segments back to original image"""
        final_image = original_image.copy()
        
        for segment in segments:
            x1, y1, x2, y2 = segment['box']
            cleaned = segment['cleaned_image']
            mask = segment['mask']
            
            cropped_mask = mask[y1:y2, x1:x2]
            
            if cropped_mask.shape[:2] != cleaned.shape[:2]:
                cropped_mask = cv2.resize(cropped_mask, (cleaned.shape[1], cleaned.shape[0]))
            
            mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
            roi = final_image[y1:y2, x1:x2]
            
            if roi.shape[:2] != cleaned.shape[:2]:
                continue
            
            final_image[y1:y2, x1:x2] = np.where(mask_3ch > 0, cleaned, roi)
        
        return final_image
    
    def run_ocr_on_segment(self, segment_image: np.ndarray) -> str:
        """
        Run OCR on một segment đã được cleaned
        
        Args:
            segment_image: RGB numpy array
        
        Returns:
            str: OCR text result
        """
        try:
            from PIL import Image
            import torch
            
            # Convert numpy to PIL
            pil_image = Image.fromarray(segment_image)
            
            # Preprocess
            pixel_values = self.ocr_processor(pil_image, return_tensors="pt").pixel_values
            
            # Generate
            with torch.no_grad():
                outputs = self.ocr_model.generate(pixel_values)
            
            # Decode
            text = self.ocr_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            return text
            
        except Exception as e:
            self.logger.error(f"OCR processing error: {e}")
            import traceback
            traceback.print_exc()
            return "[OCR ERROR]"
    
    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """Convert numpy array to QImage"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.copy()


# ========== MAIN PROCESSOR ==========

class MangaPipelineProcessor(QObject):
    """Service để xử lý full manga pipeline"""
    
    result_ready = Signal(int, QImage)  # index, final result
    ocr_result_ready = Signal(int, list)  # index, OCR texts
    progress_updated = Signal(int, int, str)  # current, total, step
    error_occurred = Signal(str)
    completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.pipeline_thread: MangaPipelineThread = None
        self.model_manager = ModelManager()
    
    def process_images(self, image_paths: List[str]):
        """Start processing full pipeline"""
        if self.pipeline_thread and self.pipeline_thread.isRunning():
            self.logger.warning("Pipeline already in progress")
            return
        
        # Get model paths
        seg_model_path = self.get_segmentation_model_path()
        text_model_path = self.get_text_detection_model_path()
        ocr_model_path = self.get_ocr_model_path()
        
        if not seg_model_path or not text_model_path:
            self.error_occurred.emit("Models not found. Please download models first.")
            return
        
        if not ocr_model_path:
            self.logger.warning("OCR model path not found - OCR will be skipped")
        
        self.pipeline_thread = MangaPipelineThread(image_paths, seg_model_path, text_model_path, ocr_model_path)
        self.pipeline_thread.result_ready.connect(self.on_result_ready)
        self.pipeline_thread.ocr_result_ready.connect(self.on_ocr_result_ready)
        self.pipeline_thread.progress_updated.connect(self.on_progress_updated)
        self.pipeline_thread.error_occurred.connect(self.on_error)
        self.pipeline_thread.finished.connect(self.on_completed)
        self.pipeline_thread.start()
    
    def get_segmentation_model_path(self) -> str:
        """Get segmentation model path"""
        base_path = self.model_manager.get_model_path()
        if not base_path:
            return ""
        
        model_path = os.path.join(base_path, "yolov8_converted.onnx")
        if not os.path.exists(model_path):
            self.logger.error(f"Segmentation model not found: {model_path}")
            return ""
        
        return model_path
    
    def get_text_detection_model_path(self) -> str:
        """Get text detection model path"""
        base_path = self.model_manager.get_model_path()
        if not base_path:
            return ""
        
        model_path = os.path.join(base_path, "comictextdetector.pt.onnx")
        if not os.path.exists(model_path):
            self.logger.error(f"Text detection model not found: {model_path}")
            return ""
        
        return model_path
    
    def get_ocr_model_path(self) -> str:
        """Get OCR model path (folder containing all OCR files)"""
        base_path = self.model_manager.get_model_path()
        if not base_path:
            self.logger.warning("Model manager base path not set")
            return ""
        
        # OCR model là thư mục chứa các file config + pytorch_model.bin
        ocr_path = base_path
        
        # Kiểm tra TẤT CẢ các file cần thiết
        required_files = [
            'config.json', 
            'preprocessor_config.json',
            'pytorch_model.bin', 
            'special_tokens_map.json',
            'tokenizer_config.json',
            'vocab.txt'
        ]
        missing_files = []
                
        for file in required_files:
            file_path = os.path.join(ocr_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
                self.logger.error(f"OCR file not found: {file_path}")
            else:
                file_size = os.path.getsize(file_path)
        
        if missing_files:
            self.logger.error(f"Missing OCR files: {missing_files}")
            self.logger.error("Please re-download OCR model files")
            return ""
        
        return ocr_path
    
    def on_ocr_result_ready(self, index: int, texts: list):
        """Forward OCR result"""
        self.ocr_result_ready.emit(index, texts)
    
    def on_result_ready(self, index: int, result_image: QImage):
        """Forward result"""
        self.result_ready.emit(index, result_image)
    
    def on_progress_updated(self, current: int, total: int, step: str):
        """Forward progress"""
        self.progress_updated.emit(current, total, step)
    
    def on_error(self, error_msg: str):
        """Forward error"""
        self.error_occurred.emit(error_msg)
    
    def on_completed(self):
        """Handle completion"""
        self.completed.emit()