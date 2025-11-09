from dataclasses import dataclass
from typing import Optional, List
from PySide6.QtGui import QImage
import numpy as np


@dataclass
class ImageData:
    """Data model cho một image trong project"""
    
    index: int
    path: str
    original_image: Optional[np.ndarray] = None
    processed_image: Optional[QImage] = None
    visualization_image: Optional[QImage] = None
    text_detection_image: Optional[QImage] = None
    
    def __post_init__(self):
        """Validate data sau khi khởi tạo"""
        if self.index < 0:
            raise ValueError("Image index must be non-negative")
        if not self.path:
            raise ValueError("Image path cannot be empty")