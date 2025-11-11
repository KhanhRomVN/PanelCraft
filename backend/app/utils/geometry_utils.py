import numpy as np
import cv2
from typing import List, Tuple, Optional

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