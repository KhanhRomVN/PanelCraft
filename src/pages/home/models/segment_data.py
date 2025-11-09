from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class RectangleData:
    """Data model cho rectangle metadata"""
    
    id: int
    x: int
    y: int
    w: int
    h: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RectangleData':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            x=data['x'],
            y=data['y'],
            w=data['w'],
            h=data['h']
        )


@dataclass
class SegmentData:
    """Data model cho một segment từ segmentation"""
    
    id: int
    box: List[int]  # [x1, y1, x2, y2]
    score: float
    mask: np.ndarray
    cropped_original: np.ndarray
    cropped_mask: np.ndarray
    rectangle: Optional[List[int]] = None  # [x, y, w, h]
    cleaned_image: Optional[np.ndarray] = None
    
    def get_rectangle_data(self) -> Optional[RectangleData]:
        """Get rectangle as RectangleData object"""
        if self.rectangle is None:
            return Nonesegment_data
        
        x, y, w, h = self.rectangle
        return RectangleData(id=self.id, x=x, y=y, w=w, h=h)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'box': self.box,
            'score': self.score,
            'rectangle': self.rectangle
        }