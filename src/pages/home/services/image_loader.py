from PySide6.QtCore import QObject, Signal, QThread
from typing import List
import logging
import os


class ImageLoaderThread(QThread):
    """Thread để load images không block UI"""
    
    images_loaded = Signal(list)  # List[str] - image paths
    error_occurred = Signal(str)
    
    def __init__(self, image_paths: List[str]):
        super().__init__()
        self.image_paths = image_paths
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Load và validate images"""
        try:
            valid_paths = []
            
            for path in self.image_paths:
                if os.path.exists(path) and os.path.isfile(path):
                    # Basic validation - check if file is readable
                    try:
                        with open(path, 'rb') as f:
                            # Read first few bytes to verify it's an image
                            header = f.read(8)
                            if self.is_valid_image_header(header):
                                valid_paths.append(path)
                            else:
                                self.logger.warning(f"Invalid image header: {path}")
                    except Exception as e:
                        self.logger.error(f"Error reading file {path}: {e}")
                else:
                    self.logger.warning(f"File not found: {path}")
            
            if valid_paths:
                self.images_loaded.emit(valid_paths)
            else:
                self.error_occurred.emit("No valid images found")
                
        except Exception as e:
            self.logger.error(f"Error in image loading thread: {e}")
            self.error_occurred.emit(str(e))
    
    def is_valid_image_header(self, header: bytes) -> bool:
        """Check if file header indicates a valid image format"""
        # JPEG
        if header[:2] == b'\xff\xd8':
            return True
        # PNG
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return True
        return False


class ImageLoader(QObject):
    """Service để load và quản lý images"""
    
    # Signals
    images_loaded = Signal(list)  # List[str]
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.loader_thread: ImageLoaderThread = None
    
    def load_images(self, image_paths: List[str]):
        """Load images trong background thread"""
        if self.loader_thread and self.loader_thread.isRunning():
            self.logger.warning("Image loading already in progress")
            return
        
        self.logger.info(f"Starting to load {len(image_paths)} images")
        
        self.loader_thread = ImageLoaderThread(image_paths)
        self.loader_thread.images_loaded.connect(self.on_images_loaded)
        self.loader_thread.error_occurred.connect(self.on_error)
        self.loader_thread.start()
    
    def on_images_loaded(self, image_paths: List[str]):
        """Handle images loaded"""
        self.logger.info(f"Successfully loaded {len(image_paths)} images")
        self.images_loaded.emit(image_paths)
    
    def on_error(self, error_msg: str):
        """Handle error"""
        self.logger.error(f"Image loading error: {error_msg}")
        self.error_occurred.emit(error_msg)