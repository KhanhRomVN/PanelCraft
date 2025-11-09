import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """
    Convert numpy array to QImage
    
    Args:
        image: numpy array (RGB format)
    
    Returns:
        QImage object
    """
    if image is None or image.size == 0:
        logger.warning("Empty image provided to numpy_to_qimage")
        return QImage()
    
    height, width = image.shape[:2]
    
    if len(image.shape) == 2:
        # Grayscale
        qimage = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
    else:
        # RGB
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    
    return qimage.copy()


def qimage_to_numpy(qimage: QImage) -> Optional[np.ndarray]:
    """
    Convert QImage to numpy array
    
    Args:
        qimage: QImage object
    
    Returns:
        numpy array (RGB format)
    """
    if qimage.isNull():
        return None
    
    width = qimage.width()
    height = qimage.height()
    
    ptr = qimage.bits()
    arr = np.array(ptr).reshape(height, width, 3)
    
    return arr


def load_image_rgb(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from path and convert to RGB
    
    Args:
        image_path: Path to image file
    
    Returns:
        numpy array (RGB format) or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def resize_with_aspect_ratio(image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image while keeping aspect ratio and add padding
    
    Args:
        image: Input image (numpy array)
        target_size: Target size (both width and height)
    
    Returns:
        Tuple of (resized_image, scale, pad_w, pad_h)
    """
    orig_h, orig_w = image.shape[:2]
    
    # Calculate scale
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return canvas, scale, pad_w, pad_h


def validate_image_file(file_path: str) -> bool:
    """
    Validate if file is a valid image
    
    Args:
        file_path: Path to file
    
    Returns:
        True if valid image, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            
            # Check JPEG
            if header[:2] == b'\xff\xd8':
                return True
            
            # Check PNG
            if header[:8] == b'\x89PNG\r\n\x1a\n':
                return True
            
            return False
    except Exception as e:
        logger.error(f"Error validating image {file_path}: {e}")
        return False