import cv2
import numpy as np
import base64
import os
from typing import Optional
import uuid

def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    try:
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Encode to JPEG
        success, encoded_image = cv2.imencode('.jpg', image_bgr)
        if success:
            return base64.b64encode(encoded_image).decode('utf-8')
        else:
            return ""
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def base64_to_numpy(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 string to numpy image"""
    try:
        image_data = base64.b64decode(base64_string)
        np_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def save_temp_image(image: np.ndarray, prefix: str = "image") -> str:
    """Save image to temporary directory and return path"""
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Generate unique filename
        filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join("temp", filename)
        
        # Convert RGB to BGR for saving
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Save image
        cv2.imwrite(filepath, image_bgr)
        
        return filepath
        
    except Exception as e:
        print(f"Error saving temp image: {e}")
        raise

def resize_image(image: np.ndarray, max_size: int = 2048) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)