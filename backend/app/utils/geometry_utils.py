import numpy as np
import cv2
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.pipeline_models import SegmentData

def find_largest_inscribed_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find largest inscribed rectangle in binary mask
    """
    if mask.sum() == 0:
        return None
    
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
    h, w = mask_crop.shape
    
    max_area = 0
    best_rect = None
    
    # Build height map
    height_map = np.zeros((h, w), dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            if mask_crop[i, j] > 0:
                if i == 0:
                    height_map[i, j] = 1
                else:
                    height_map[i, j] = height_map[i-1, j] + 1
            else:
                height_map[i, j] = 0
    
    # Find largest rectangle in each histogram
    for i in range(h):
        histogram = height_map[i, :]
        rect = largest_rectangle_in_histogram(histogram, i)
        
        if rect is not None:
            x, y, rw, rh = rect
            area = rw * rh
            
            if is_rectangle_inside_mask(mask_crop, x, y, rw, rh):
                if area > max_area:
                    max_area = area
                    best_rect = (x + x_min, y + y_min, rw, rh)
    
    return best_rect

def largest_rectangle_in_histogram(histogram: np.ndarray, row_index: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Find largest rectangle in histogram
    """
    stack = []
    max_area = 0
    best_rect = None
    index = 0
    
    while index < len(histogram):
        if not stack or histogram[index] >= histogram[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            top = stack.pop()
            height = histogram[top]
            width = index if not stack else index - stack[-1] - 1
            area = height * width
            
            if area > max_area:
                max_area = area
                x = stack[-1] + 1 if stack else 0
                y = row_index - height + 1
                best_rect = (x, y, width, height)
    
    while stack:
        top = stack.pop()
        height = histogram[top]
        width = index if not stack else index - stack[-1] - 1
        area = height * width
        
        if area > max_area:
            max_area = area
            x = stack[-1] + 1 if stack else 0
            y = row_index - height + 1
            best_rect = (x, y, width, height)
    
    return best_rect

def is_rectangle_inside_mask(mask: np.ndarray, x: int, y: int, width: int, height: int) -> bool:
    """
    Check if rectangle is completely inside mask
    """
    if x < 0 or y < 0 or x + width > mask.shape[1] or y + height > mask.shape[0]:
        return False
    
    rect_region = mask[y:y+height, x:x+width]
    return np.all(rect_region > 0)

def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Apply Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return []
    
    # Convert from center format to corner format
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
        
        # Calculate IoU
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
        
        # Keep boxes with IoU < threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def filter_segments_by_quality(
    segments: List['SegmentData'],
    std_threshold: float = 2.0,
    min_area: int = 100
) -> Tuple[List['SegmentData'], dict]:
    """
    Filter segments dựa trên standard deviation của area và aspect ratio
    
    Args:
        segments: List segments cần filter
        std_threshold: Ngưỡng standard deviation (default: 2.0)
        min_area: Diện tích tối thiểu (pixels)
    
    Returns:
        Tuple[List[SegmentData], dict]: (filtered_segments, statistics)
    """
    if len(segments) == 0:
        return [], {}
    
    # Extract metrics
    areas = []
    aspect_ratios = []
    
    for seg in segments:
        x1, y1, x2, y2 = seg.box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area < min_area:
            continue
        
        areas.append(area)
        aspect_ratio = width / max(height, 1)
        aspect_ratios.append(aspect_ratio)
    
    if len(areas) == 0:
        return [], {}
    
    # Calculate statistics
    area_mean = np.mean(areas)
    area_std = np.std(areas)
    ar_mean = np.mean(aspect_ratios)
    ar_std = np.std(aspect_ratios)
    
    # Calculate thresholds
    area_min = max(min_area, area_mean - std_threshold * area_std)
    area_max = area_mean + std_threshold * area_std
    ar_min = max(0.1, ar_mean - std_threshold * ar_std)
    ar_max = min(10.0, ar_mean + std_threshold * ar_std)
    
    # Filter segments
    filtered = []
    rejected_count = 0
    
    for seg in segments:
        x1, y1, x2, y2 = seg.box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # Check conditions
        if area < area_min or area > area_max:
            rejected_count += 1
            continue
        
        if aspect_ratio < ar_min or aspect_ratio > ar_max:
            rejected_count += 1
            continue
        
        filtered.append(seg)
    
    # Statistics
    stats = {
        "original_count": len(segments),
        "filtered_count": len(filtered),
        "rejected_count": rejected_count,
        "area_mean": float(area_mean),
        "area_std": float(area_std),
        "area_range": [float(area_min), float(area_max)],
        "aspect_ratio_mean": float(ar_mean),
        "aspect_ratio_std": float(ar_std),
        "aspect_ratio_range": [float(ar_min), float(ar_max)]
    }
    
    return filtered, stats

def filter_text_boxes_by_quality(
    boxes: np.ndarray,
    scores: np.ndarray,
    std_threshold: float = 2.0,
    min_area: int = 50,
    min_score: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Filter text boxes dựa trên standard deviation và confidence score
    
    Args:
        boxes: Array of boxes [x1, y1, x2, y2]
        scores: Confidence scores
        std_threshold: Ngưỡng standard deviation
        min_area: Diện tích tối thiểu
        min_score: Score tối thiểu
    
    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: (filtered_boxes, filtered_scores, statistics)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([]), {}
    
    # Filter by score first
    valid_mask = scores >= min_score
    if not np.any(valid_mask):
        return np.array([]), np.array([]), {}
    
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    
    # Extract metrics
    areas = []
    aspect_ratios = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if area < min_area:
            continue
        
        areas.append(area)
        aspect_ratio = width / max(height, 1)
        aspect_ratios.append(aspect_ratio)
    
    if len(areas) == 0:
        return np.array([]), np.array([]), {}
    
    # Calculate statistics
    area_mean = np.mean(areas)
    area_std = np.std(areas)
    ar_mean = np.mean(aspect_ratios)
    ar_std = np.std(aspect_ratios)
    
    # Calculate thresholds
    area_min = max(min_area, area_mean - std_threshold * area_std)
    area_max = area_mean + std_threshold * area_std
    ar_min = max(0.1, ar_mean - std_threshold * ar_std)
    ar_max = min(10.0, ar_mean + std_threshold * ar_std)
    
    # Filter boxes
    filtered_boxes = []
    filtered_scores = []
    rejected_count = 0
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        if area < area_min or area > area_max:
            rejected_count += 1
            continue
        
        if aspect_ratio < ar_min or aspect_ratio > ar_max:
            rejected_count += 1
            continue
        
        filtered_boxes.append(box)
        filtered_scores.append(score)
    
    filtered_boxes = np.array(filtered_boxes) if filtered_boxes else np.array([])
    filtered_scores = np.array(filtered_scores) if filtered_scores else np.array([])
    
    # Statistics
    stats = {
        "original_count": len(boxes),
        "filtered_count": len(filtered_boxes),
        "rejected_count": rejected_count,
        "area_mean": float(area_mean),
        "area_std": float(area_std),
        "area_range": [float(area_min), float(area_max)],
        "aspect_ratio_mean": float(ar_mean),
        "aspect_ratio_std": float(ar_std),
        "aspect_ratio_range": [float(ar_min), float(ar_max)]
    }
    
    return filtered_boxes, filtered_scores, stats