from typing import TypedDict, List, Optional
from PySide6.QtGui import QImage
import numpy as np


class RectangleDict(TypedDict):
    """Type definition for rectangle dictionary"""
    id: int
    x: int
    y: int
    w: int
    h: int


class SegmentDict(TypedDict):
    """Type definition for segment dictionary"""
    id: int
    box: List[int]
    score: float
    mask: np.ndarray
    cropped_original: np.ndarray
    cropped_mask: np.ndarray
    rectangle: Optional[List[int]]
    cleaned_image: Optional[np.ndarray]


class ImageResultDict(TypedDict):
    """Type definition for image processing result"""
    image: QImage
    rectangles: List[RectangleDict]


class OCRTableData(TypedDict):
    """Type definition for OCR table row data"""
    STT: str
    Character: str
    original_text: str
    Translation: str
    _full_original: str
    _full_translation: str
    _character_id: Optional[int]


class ProcessingProgress(TypedDict):
    """Type definition for processing progress"""
    current: int
    total: int
    step: str