import logging
import os
from PIL import Image
from PySide6.QtCore import QObject, Signal, QThread
from core.model_manager import ModelManager

class OCRThread(QThread):
    result_ready = Signal(int, list)  # index, list of text
    progress_updated = Signal(int, int)  # current, total
    error_occurred = Signal(str)

    def __init__(self, image_paths: list, model_path: str):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
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

class OCRProcessor(QObject):
    """Service để xử lý OCR"""
    
    result_ready = Signal(int, list)  # index, list of text
    progress_updated = Signal(int, int)
    error_occurred = Signal(str)
    completed = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.ocr_thread: OCRThread = None
        self.model_manager = ModelManager()
    
    def process_images(self, image_paths: list):
        """Bắt đầu xử lý OCR"""
        if self.ocr_thread and self.ocr_thread.isRunning():
            self.logger.warning("OCR already in progress")
            return
        
        # OCR model doesn't need explicit path as manga-ocr handles it        
        self.ocr_thread = OCRThread(image_paths, "")
        self.ocr_thread.result_ready.connect(self.on_result_ready)
        self.ocr_thread.progress_updated.connect(self.on_progress_updated)
        self.ocr_thread.error_occurred.connect(self.on_error)
        self.ocr_thread.finished.connect(self.on_completed)
        self.ocr_thread.start()
    
    def on_result_ready(self, index: int, texts: list):
        """Forward result signal"""
        self.result_ready.emit(index, texts)
    
    def on_progress_updated(self, current: int, total: int):
        """Forward progress signal"""
        self.progress_updated.emit(current, total)
    
    def on_error(self, error_msg: str):
        """Forward error signal"""
        self.error_occurred.emit(error_msg)
    
    def on_completed(self):
        """Handle completion"""
        self.completed.emit()