import logging
import os
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QImage
from core.model_manager import ModelManager
import onnxruntime as ort

class TextDetectionThread(QThread):
    result_ready = Signal(int, QImage)  # index, result image
    progress_updated = Signal(int, int)  # current, total
    error_occurred = Signal(str)

    def __init__(self, image_paths: list, model_path: str):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        self.session = None

    def run(self):
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
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
                    continue
                        
        except Exception as e:
            self.logger.error(f"Error in text detection thread: {e}")
            self.error_occurred.emit(f"Text detection error: {str(e)}")

    def detect_text(self, image: np.ndarray) -> np.ndarray:
        """Run comic text detection on image"""
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_tensor})
        
        # Postprocess - draw bounding boxes around detected text
        result = self.postprocess(outputs, image)
        
        return result

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for text detection"""
        # Resize to model input size (assuming 1024x1024)
        input_size = 1024
        resized = cv2.resize(image, (input_size, input_size))
        
        # Normalize and convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched

    def postprocess(self, outputs: list, original_image: np.ndarray) -> np.ndarray:
        """Postprocess text detection results - draw bounding boxes"""
        # Convert to RGB for display
        result_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Simple bounding box drawing (you'll need to adapt this based on actual model output)
        # This is a simplified version - you'll need to parse the actual model outputs
        height, width = original_image.shape[:2]
        
        # Draw some example boxes (replace with actual model output parsing)
        # In practice, you'd parse the model outputs to get actual text regions
        cv2.rectangle(result_image, (50, 50), (200, 100), (0, 255, 0), 2)
        cv2.rectangle(result_image, (300, 150), (500, 200), (0, 255, 0), 2)
        
        # Add text labels
        cv2.putText(result_image, "Detected Text", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image

    def numpy_to_qimage(self, image: np.ndarray) -> QImage:
        """Convert numpy array to QImage"""
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return qimage.copy()

class TextDetectionProcessor(QObject):
    """Service để xử lý text detection"""
    
    result_ready = Signal(int, QImage)
    progress_updated = Signal(int, int)
    error_occurred = Signal(str)
    completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.text_detection_thread: TextDetectionThread = None
        self.model_manager = ModelManager()
    
    def process_images(self, image_paths: list):
        """Bắt đầu xử lý text detection"""
        if self.text_detection_thread and self.text_detection_thread.isRunning():
            self.logger.warning("Text detection already in progress")
            return
        
        # Get model path
        model_path = self.get_model_path()
        if not model_path:
            self.error_occurred.emit("Text detection model not found. Please download models first.")
            return
                
        self.text_detection_thread = TextDetectionThread(image_paths, model_path)
        self.text_detection_thread.result_ready.connect(self.on_result_ready)
        self.text_detection_thread.progress_updated.connect(self.on_progress_updated)
        self.text_detection_thread.error_occurred.connect(self.on_error)
        self.text_detection_thread.finished.connect(self.on_completed)
        self.text_detection_thread.start()
    
    def get_model_path(self) -> str:
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
        self.completed.emit()