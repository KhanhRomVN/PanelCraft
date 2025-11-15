"""
Image Processing Utilities (Authoritative Module)

Responsibilities:
- Provide reusable image conversion, resizing, and temporary persistence helpers.
- Centralize logic previously duplicated in legacy app/utils/image_utils.py.
- Integrate application settings (TEMP_DIR, MAX_IMAGE_SIZE) & structured logging.
- Offer safe base64 encode/decode utilities for lightweight transport.

Functions:
    numpy_to_base64(image) -> str
    base64_to_numpy(data) -> np.ndarray | None
    save_temp_image(image, prefix="image") -> str
    resize_image(image, max_size=settings.MAX_IMAGE_SIZE) -> np.ndarray

Refactor Improvements:
- Added type hints & comprehensive docstrings.
- Use settings.TEMP_DIR instead of hard-coded "temp".
- Added logging and graceful error handling (no silent prints).
- Ensured RGB/BGR conversion parity.
- Prepared for future extension (e.g., async filesystem, storage backends).

NOTE:
Legacy module app/utils/image_utils.py now acts as a deprecated shim
re-exporting these functions. Import from this module moving forward.

"""

from __future__ import annotations

import os
import uuid
import base64
import logging
from typing import Optional

import cv2
import numpy as np

from app.config.settings import settings

logger = logging.getLogger(__name__)


def numpy_to_base64(image: np.ndarray) -> str:
    """
    Encode an RGB numpy image into a base64 JPEG string.

    Args:
        image: np.ndarray image (RGB or grayscale)

    Returns:
        Base64 string of JPEG-encoded image. Empty string if encoding fails.
    """
    try:
        if image is None or image.size == 0:
            logger.warning("[image_utils] numpy_to_base64 received empty image")
            return ""

        # Convert RGB -> BGR for OpenCV JPEG encoding if 3-channel
        if image.ndim == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        success, encoded = cv2.imencode(".jpg", image_bgr)
        if not success:
            logger.error("[image_utils] JPEG encoding failed")
            return ""

        return base64.b64encode(encoded).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        logger.error(f"[image_utils] numpy_to_base64 error: {e}")
        return ""


def base64_to_numpy(base64_string: str) -> Optional[np.ndarray]:
    """
    Decode a base64 JPEG/PNG string into an RGB np.ndarray.

    Args:
        base64_string: Base64 encoded image

    Returns:
        RGB np.ndarray or None on failure.
    """
    try:
        if not base64_string:
            return None

        image_data = base64.b64decode(base64_string)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            logger.error("[image_utils] base64_to_numpy imdecode returned None")
            return None

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[image_utils] base64_to_numpy error: {e}")
        return None


def save_temp_image(image: np.ndarray, prefix: str = "image") -> str:
    """
    Persist an image to the temporary directory.

    Args:
        image: RGB or BGR image
        prefix: filename prefix (logical category / purpose)

    Returns:
        Absolute file path to saved image.

    Raises:
        RuntimeError if save fails.
    """
    try:
        if image is None or image.size == 0:
            raise ValueError("Cannot save empty image")

        temp_dir = os.path.abspath(settings.TEMP_DIR)
        os.makedirs(temp_dir, exist_ok=True)

        filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(temp_dir, filename)

        # Ensure BGR for OpenCV write
        if image.ndim == 3 and image.shape[2] == 3:
            # Heuristic: assume incoming is RGB (pipeline uses RGB internally)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        if not cv2.imwrite(filepath, image_bgr):
            raise RuntimeError(f"cv2.imwrite failed for path: {filepath}")

        return filepath
    except Exception as e:  # noqa: BLE001
        logger.error(f"[image_utils] save_temp_image error: {e}")
        raise


def resize_image(image: np.ndarray, max_size: int = settings.MAX_IMAGE_SIZE) -> np.ndarray:
    """
    Resize an image to fit within max_size while preserving aspect ratio.

    Args:
        image: Input RGB/BGR image
        max_size: Maximum width or height boundary

    Returns:
        Resized image (np.ndarray) or original if already within limits.
    """
    try:
        if image is None or image.size == 0:
            logger.warning("[image_utils] resize_image received empty image")
            return image

        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image

        if h >= w:
            new_h = max_size
            new_w = int(w * max_size / max(h, 1))
        else:
            new_w = max_size
            new_h = int(h * max_size / max(w, 1))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(
            "[image_utils] Resized image from %dx%d to %dx%d (max_size=%d)",
            w,
            h,
            new_w,
            new_h,
            max_size,
        )
        return resized
    except Exception as e:  # noqa: BLE001
        logger.error(f"[image_utils] resize_image error: {e}")
        return image


__all__ = [
    "numpy_to_base64",
    "base64_to_numpy",
    "save_temp_image",
    "resize_image",
]
