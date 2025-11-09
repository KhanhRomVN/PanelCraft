from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                              QFileDialog, QScrollArea, QMessageBox, QPushButton, QSplitter)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QKeyEvent, QImage
import logging
import os
from pathlib import Path
from typing import List, Optional

from widget.common.custom_button import CustomButton
from widget.common.custom_modal import CustomModal
from .interactive_image_label import InteractiveImageLabel
from ...services.image_loader import ImageLoader
from ...services.segmentation_processor import SegmentationProcessor
from ...services.manga_pipeline_processor import MangaPipelineProcessor
from ...constants.constants import VALID_IMAGE_EXTENSIONS


class ImageDisplayWidget(QWidget):
    """Widget hiển thị ảnh với scroll và rectangles có thể drag"""
    
    def __init__(self, title: str = ""):
        super().__init__()
        self.current_pixmap: Optional[QPixmap] = None
        self.rectangles: List[dict] = []
        self.ocr_mode_enabled: bool = False  # CHẾ ĐỘ OCR DRAG
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: var(--text-primary);
                padding: 8px;
                background-color: var(--sidebar-background);
                border-radius: 4px;
            """)
            layout.addWidget(title_label)
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid var(--border);
                border-radius: 4px;
                background-color: var(--card-background);
            }
        """)
        
        # Image label - use InteractiveImageLabel for drag-drop rectangles
        self.image_label = InteractiveImageLabel()
        self.image_label.setText("No image loaded")
        
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
    
    def set_image(self, image_path: str = None, pixmap: QPixmap = None, rectangles: List[dict] = None):
        """Set image to display với optional rectangles"""
        if pixmap:
            self.current_pixmap = pixmap
        elif image_path and os.path.exists(image_path):
            self.current_pixmap = QPixmap(image_path)
        else:
            self.current_pixmap = None
        
        # Cập nhật rectangles
        if rectangles is not None:
            self.rectangles = rectangles
            self.image_label.set_rectangles(rectangles)
        
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.image_label.setPixmap(self.current_pixmap)
        else:
            self.image_label.setText("Failed to load image")
    
    def clear(self):
        """Clear image"""
        self.current_pixmap = None
        self.rectangles.clear()
        self.image_label.clear()
        self.image_label.set_rectangles([])
        self.image_label.setText("No image loaded")
    
    def get_rectangles(self) -> List[dict]:
        """Lấy danh sách rectangles hiện tại (sau khi drag)"""
        return self.image_label.get_rectangles()
    
    def on_ocr_region_selected(self, x: int, y: int, w: int, h: int):
        """Handle khi user chọn vùng OCR bằng drag"""
        # Emit signal lên CanvasPanel để xử lý
        if hasattr(self.parent(), 'on_ocr_region_selected'):
            self.parent().on_ocr_region_selected(x, y, w, h)
    
    def enable_ocr_drag(self):
        """Bật chế độ OCR drag"""
        self._ocr_mode = True
        self.setCursor(Qt.CrossCursor)
    
    def disable_ocr_drag(self):
        """Tắt chế độ OCR drag"""
        self._ocr_mode = False
        self._ocr_drag_rect = None
        self.setCursor(Qt.ArrowCursor)
        self.updateDisplay()
    
class InteractiveImageLabel(QLabel):
    """QLabel với khả năng drag-and-drop và resize rectangles"""
    
    # Resize modes
    RESIZE_NONE = 0
    RESIZE_TOP_LEFT = 1
    RESIZE_TOP = 2
    RESIZE_TOP_RIGHT = 3
    RESIZE_RIGHT = 4
    RESIZE_BOTTOM_RIGHT = 5
    RESIZE_BOTTOM = 6
    RESIZE_BOTTOM_LEFT = 7
    RESIZE_LEFT = 8
    MOVE = 9
    
    def __init__(self):
        super().__init__()
        self._pixmap: Optional[QPixmap] = None
        self._rectangles: List[dict] = []
        self._active_rect_id: Optional[int] = None
        self._resize_mode: int = self.RESIZE_NONE
        self._drag_start_pos = None
        self._rect_original_geometry = None
        self._handle_size = 8
        
        # OCR drag mode
        self._ocr_mode: bool = False
        self._ocr_drag_start = None
        self._ocr_drag_rect = None
        
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("""
            QLabel {
                background-color: var(--card-background);
                padding: 8px;
            }
        """)
    
    def setPixmap(self, pixmap: QPixmap):
        """Override setPixmap"""
        if pixmap:
            self._pixmap = pixmap
            self.updateDisplay()
        else:
            self._pixmap = None
            super().setPixmap(QPixmap())
    
    def set_rectangles(self, rectangles: List[dict]):
        """Set danh sách rectangles"""
        self._rectangles = rectangles.copy() if rectangles else []
        self.updateDisplay()
    
    def get_rectangles(self) -> List[dict]:
        """Get danh sách rectangles hiện tại"""
        return self._rectangles.copy()
    
    def enable_ocr_mode(self):
        """Bật chế độ OCR drag-and-drop"""
        self.ocr_mode_enabled = True
        self.image_label.enable_ocr_drag()
    
    def disable_ocr_mode(self):
        """Tắt chế độ OCR drag-and-drop"""
        self.ocr_mode_enabled = False
        self.image_label.disable_ocr_drag()
    
    def resizeEvent(self, event):
        """Auto resize khi label resize"""
        super().resizeEvent(event)
        if self._pixmap:
            self.updateDisplay()
    
    def updateDisplay(self):
        """Vẽ pixmap + rectangles"""
        if not self._pixmap or self._pixmap.isNull():
            return
        
        # Scale pixmap
        available_width = self.width() - 16
        available_height = self.height() - 16
        
        if available_width <= 0 or available_height <= 0:
            return
        
        scaled_pixmap = self._pixmap.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Vẽ rectangles lên pixmap
        if self._rectangles:
            from PySide6.QtGui import QPainter, QPen, QBrush
            from PySide6.QtCore import QRect, QRectF
            
            # Copy pixmap để vẽ
            display_pixmap = scaled_pixmap.copy()
            painter = QPainter(display_pixmap)
            
            # Calculate scale ratio
            scale_x = scaled_pixmap.width() / self._pixmap.width()
            scale_y = scaled_pixmap.height() / self._pixmap.height()
            
            # Vẽ từng rectangle
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)  # QUAN TRỌNG: Không fill rectangle
            
            for idx, rect in enumerate(self._rectangles):
                # Scale coordinates
                x = int(rect['x'] * scale_x)
                y = int(rect['y'] * scale_y)
                w = int(rect['w'] * scale_x)
                h = int(rect['h'] * scale_y)
                
                painter.drawRect(QRect(x, y, w, h))
                
                # Vẽ số thứ tự ở góc phải trên (ngoài cùng rectangle)
                from PySide6.QtGui import QFont
                number_text = str(idx + 1)
                font = QFont()
                font.setPointSize(12)
                font.setBold(True)
                painter.setFont(font)
                
                # Tính kích thước text để vẽ background
                from PySide6.QtGui import QFontMetrics
                fm = QFontMetrics(font)
                text_width = fm.horizontalAdvance(number_text)
                text_height = fm.height()
                
                # Vị trí: góc phải trên, ngoài cùng rectangle
                number_x = x + w + 5
                number_y = y - 5
                
                # Vẽ background cho số (hình chữ nhật nhỏ)
                painter.setBrush(QBrush(Qt.red))
                painter.setPen(Qt.NoPen)
                padding = 4
                painter.drawRect(
                    number_x - padding,
                    number_y - text_height - padding,
                    text_width + 2 * padding,
                    text_height + 2 * padding
                )
                
                # Vẽ text số màu trắng
                painter.setPen(QPen(Qt.white))
                painter.drawText(number_x, number_y - padding, number_text)
                
                # Reset pen và brush
                painter.setPen(QPen(Qt.red, 2))
                painter.setBrush(Qt.NoBrush)
                
                # Vẽ handles nếu đây là rect đang active
                if rect['id'] == self._active_rect_id:
                    handle_size = self._handle_size
                    painter.setBrush(QBrush(Qt.red))  # CHỈ fill handles
                    
                    # 4 góc
                    painter.drawRect(x - handle_size//2, y - handle_size//2, handle_size, handle_size)  # Top-left
                    painter.drawRect(x + w - handle_size//2, y - handle_size//2, handle_size, handle_size)  # Top-right
                    painter.drawRect(x + w - handle_size//2, y + h - handle_size//2, handle_size, handle_size)  # Bottom-right
                    painter.drawRect(x - handle_size//2, y + h - handle_size//2, handle_size, handle_size)  # Bottom-left
                    
                    # 4 cạnh (mid points)
                    painter.drawRect(x + w//2 - handle_size//2, y - handle_size//2, handle_size, handle_size)  # Top
                    painter.drawRect(x + w - handle_size//2, y + h//2 - handle_size//2, handle_size, handle_size)  # Right
                    painter.drawRect(x + w//2 - handle_size//2, y + h - handle_size//2, handle_size, handle_size)  # Bottom
                    painter.drawRect(x - handle_size//2, y + h//2 - handle_size//2, handle_size, handle_size)  # Left
                    
                    painter.setBrush(Qt.NoBrush)  # QUAN TRỌNG: Reset brush sau khi vẽ handles
            
            painter.end()
            
            # Vẽ OCR drag rectangle nếu đang drag
            if self._ocr_drag_rect:
                painter_ocr = QPainter(display_pixmap)
                painter_ocr.setPen(QPen(Qt.blue, 3, Qt.DashLine))
                painter_ocr.setBrush(Qt.NoBrush)
                
                x, y, w, h = self._ocr_drag_rect
                painter_ocr.drawRect(int(x), int(y), int(w), int(h))
                
                painter_ocr.end()
            
            super().setPixmap(display_pixmap)
        else:
            # Vẽ OCR drag rectangle nếu không có rectangles nhưng đang drag OCR
            if self._ocr_drag_rect:
                display_pixmap = scaled_pixmap.copy()
                painter_ocr = QPainter(display_pixmap)
                painter_ocr.setPen(QPen(Qt.blue, 3, Qt.DashLine))
                painter_ocr.setBrush(Qt.NoBrush)
                
                x, y, w, h = self._ocr_drag_rect
                painter_ocr.drawRect(int(x), int(y), int(w), int(h))
                
                painter_ocr.end()
                super().setPixmap(display_pixmap)
            else:
                super().setPixmap(scaled_pixmap)
    
    def _get_resize_mode(self, orig_x: int, orig_y: int, rect: dict) -> int:
        """Xác định resize mode dựa trên vị trí click"""
        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
        tolerance = self._handle_size
        
        # Check corners first (priority)
        if abs(orig_x - x) <= tolerance and abs(orig_y - y) <= tolerance:
            return self.RESIZE_TOP_LEFT
        if abs(orig_x - (x + w)) <= tolerance and abs(orig_y - y) <= tolerance:
            return self.RESIZE_TOP_RIGHT
        if abs(orig_x - (x + w)) <= tolerance and abs(orig_y - (y + h)) <= tolerance:
            return self.RESIZE_BOTTOM_RIGHT
        if abs(orig_x - x) <= tolerance and abs(orig_y - (y + h)) <= tolerance:
            return self.RESIZE_BOTTOM_LEFT
        
        # Check edges
        if abs(orig_y - y) <= tolerance and x <= orig_x <= x + w:
            return self.RESIZE_TOP
        if abs(orig_x - (x + w)) <= tolerance and y <= orig_y <= y + h:
            return self.RESIZE_RIGHT
        if abs(orig_y - (y + h)) <= tolerance and x <= orig_x <= x + w:
            return self.RESIZE_BOTTOM
        if abs(orig_x - x) <= tolerance and y <= orig_y <= y + h:
            return self.RESIZE_LEFT
        
        # Inside rectangle = move
        if x <= orig_x <= x + w and y <= orig_y <= y + h:
            return self.MOVE
        
        return self.RESIZE_NONE
    
    def _get_cursor_for_mode(self, mode: int):
        """Trả về cursor phù hợp với resize mode"""
        cursor_map = {
            self.RESIZE_TOP_LEFT: Qt.SizeFDiagCursor,
            self.RESIZE_TOP: Qt.SizeVerCursor,
            self.RESIZE_TOP_RIGHT: Qt.SizeBDiagCursor,
            self.RESIZE_RIGHT: Qt.SizeHorCursor,
            self.RESIZE_BOTTOM_RIGHT: Qt.SizeFDiagCursor,
            self.RESIZE_BOTTOM: Qt.SizeVerCursor,
            self.RESIZE_BOTTOM_LEFT: Qt.SizeBDiagCursor,
            self.RESIZE_LEFT: Qt.SizeHorCursor,
            self.MOVE: Qt.OpenHandCursor,
        }
        return cursor_map.get(mode, Qt.ArrowCursor)
    
    def mousePressEvent(self, event):
        """Bắt đầu drag/resize rectangle HOẶC OCR drag"""
        if not self._pixmap or event.button() != Qt.LeftButton:
            return
        
        # Nếu đang ở OCR mode, bắt đầu drag rectangle mới
        if self._ocr_mode:
            click_pos = event.pos()
            pixmap_rect = self.pixmap().rect()
            
            offset_x = (self.width() - pixmap_rect.width()) // 2
            offset_y = (self.height() - pixmap_rect.height()) // 2
            
            rel_x = click_pos.x() - offset_x
            rel_y = click_pos.y() - offset_y
            
            if 0 <= rel_x <= pixmap_rect.width() and 0 <= rel_y <= pixmap_rect.height():
                # Lưu vị trí bắt đầu drag (tọa độ scaled)
                self._ocr_drag_start = (rel_x, rel_y)
                self._ocr_drag_rect = None
            return
        
        # Get click position relative to scaled pixmap
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        # Center alignment offset
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y
        
        # Check if click is on pixmap
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap_rect.width() or rel_y > pixmap_rect.height():
            return
        
        # Calculate scale
        scale_x = self._pixmap.width() / pixmap_rect.width()
        scale_y = self._pixmap.height() / pixmap_rect.height()
        
        # Convert to original coordinates
        orig_x = int(rel_x * scale_x)
        orig_y = int(rel_y * scale_y)
        
        # Find clicked rectangle và xác định resize mode
        for rect in reversed(self._rectangles):
            mode = self._get_resize_mode(orig_x, orig_y, rect)
            
            if mode != self.RESIZE_NONE:
                self._active_rect_id = rect['id']
                self._resize_mode = mode
                self._drag_start_pos = (orig_x, orig_y)
                self._rect_original_geometry = (rect['x'], rect['y'], rect['w'], rect['h'])
                
                # Set cursor
                if mode == self.MOVE:
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    self.setCursor(self._get_cursor_for_mode(mode))
                
                # Trigger repaint để hiển thị handles
                self.updateDisplay()
                break
    
    def mouseMoveEvent(self, event):
        """Di chuyển/resize rectangle HOẶC vẽ OCR drag rectangle"""
        if not self._pixmap:
            return
        
        # Xử lý OCR drag mode
        if self._ocr_mode and self._ocr_drag_start:
            click_pos = event.pos()
            pixmap_rect = self.pixmap().rect()
            
            offset_x = (self.width() - pixmap_rect.width()) // 2
            offset_y = (self.height() - pixmap_rect.height()) // 2
            
            rel_x = click_pos.x() - offset_x
            rel_y = click_pos.y() - offset_y
            
            # Clamp coordinates
            rel_x = max(0, min(rel_x, pixmap_rect.width()))
            rel_y = max(0, min(rel_y, pixmap_rect.height()))
            
            # Tạo rectangle từ start đến current
            start_x, start_y = self._ocr_drag_start
            
            x = min(start_x, rel_x)
            y = min(start_y, rel_y)
            w = abs(rel_x - start_x)
            h = abs(rel_y - start_y)
            
            self._ocr_drag_rect = (x, y, w, h)
            self.updateDisplay()
            return
        
        # Get position
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y
        
        # Check bounds
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap_rect.width() or rel_y > pixmap_rect.height():
            if self._resize_mode == self.RESIZE_NONE:
                self.setCursor(Qt.ArrowCursor)
            return
        
        # Calculate scale
        scale_x = self._pixmap.width() / pixmap_rect.width()
        scale_y = self._pixmap.height() / pixmap_rect.height()
        
        # Convert to original coordinates
        orig_x = int(rel_x * scale_x)
        orig_y = int(rel_y * scale_y)
        
        # If not dragging/resizing, update cursor
        if self._resize_mode == self.RESIZE_NONE:
            for rect in reversed(self._rectangles):
                mode = self._get_resize_mode(orig_x, orig_y, rect)
                if mode != self.RESIZE_NONE:
                    self.setCursor(self._get_cursor_for_mode(mode))
                    return
            self.setCursor(Qt.ArrowCursor)
            return
        
        # Apply drag/resize
        for rect in self._rectangles:
            if rect['id'] == self._active_rect_id:
                dx = orig_x - self._drag_start_pos[0]
                dy = orig_y - self._drag_start_pos[1]
                
                orig_x_rect, orig_y_rect, orig_w_rect, orig_h_rect = self._rect_original_geometry
                
                min_size = 10  # Kích thước tối thiểu
                
                if self._resize_mode == self.MOVE:
                    # Move rectangle
                    new_x = orig_x_rect + dx
                    new_y = orig_y_rect + dy
                    rect['x'] = max(0, min(new_x, self._pixmap.width() - rect['w']))
                    rect['y'] = max(0, min(new_y, self._pixmap.height() - rect['h']))
                    
                elif self._resize_mode == self.RESIZE_TOP_LEFT:
                    new_x = max(0, min(orig_x_rect + dx, orig_x_rect + orig_w_rect - min_size))
                    new_y = max(0, min(orig_y_rect + dy, orig_y_rect + orig_h_rect - min_size))
                    rect['x'] = new_x
                    rect['y'] = new_y
                    rect['w'] = orig_x_rect + orig_w_rect - new_x
                    rect['h'] = orig_y_rect + orig_h_rect - new_y
                    
                elif self._resize_mode == self.RESIZE_TOP:
                    new_y = max(0, min(orig_y_rect + dy, orig_y_rect + orig_h_rect - min_size))
                    rect['y'] = new_y
                    rect['h'] = orig_y_rect + orig_h_rect - new_y
                    
                elif self._resize_mode == self.RESIZE_TOP_RIGHT:
                    new_y = max(0, min(orig_y_rect + dy, orig_y_rect + orig_h_rect - min_size))
                    new_w = max(min_size, min(orig_w_rect + dx, self._pixmap.width() - orig_x_rect))
                    rect['y'] = new_y
                    rect['w'] = new_w
                    rect['h'] = orig_y_rect + orig_h_rect - new_y
                    
                elif self._resize_mode == self.RESIZE_RIGHT:
                    rect['w'] = max(min_size, min(orig_w_rect + dx, self._pixmap.width() - orig_x_rect))
                    
                elif self._resize_mode == self.RESIZE_BOTTOM_RIGHT:
                    rect['w'] = max(min_size, min(orig_w_rect + dx, self._pixmap.width() - orig_x_rect))
                    rect['h'] = max(min_size, min(orig_h_rect + dy, self._pixmap.height() - orig_y_rect))
                    
                elif self._resize_mode == self.RESIZE_BOTTOM:
                    rect['h'] = max(min_size, min(orig_h_rect + dy, self._pixmap.height() - orig_y_rect))
                    
                elif self._resize_mode == self.RESIZE_BOTTOM_LEFT:
                    new_x = max(0, min(orig_x_rect + dx, orig_x_rect + orig_w_rect - min_size))
                    rect['x'] = new_x
                    rect['w'] = orig_x_rect + orig_w_rect - new_x
                    rect['h'] = max(min_size, min(orig_h_rect + dy, self._pixmap.height() - orig_y_rect))
                    
                elif self._resize_mode == self.RESIZE_LEFT:
                    new_x = max(0, min(orig_x_rect + dx, orig_x_rect + orig_w_rect - min_size))
                    rect['x'] = new_x
                    rect['w'] = orig_x_rect + orig_w_rect - new_x
                
                break
        
        self.updateDisplay()
    
    def mouseReleaseEvent(self, event):
        """Kết thúc drag/resize HOẶC OCR drag"""
        if event.button() == Qt.LeftButton:
            # Xử lý OCR drag mode
            if self._ocr_mode and self._ocr_drag_start and self._ocr_drag_rect:
                x, y, w, h = self._ocr_drag_rect
                
                # Chỉ xử lý nếu rectangle đủ lớn (tối thiểu 20x20 pixels)
                if w >= 20 and h >= 20:
                    # Convert scaled coordinates về original image coordinates
                    if self._pixmap:
                        pixmap_rect = self.pixmap().rect()
                        scale_x = self._pixmap.width() / pixmap_rect.width()
                        scale_y = self._pixmap.height() / pixmap_rect.height()
                        
                        orig_x = int(x * scale_x)
                        orig_y = int(y * scale_y)
                        orig_w = int(w * scale_x)
                        orig_h = int(h * scale_y)
                        
                        # Emit signal để parent widget xử lý OCR
                        from PySide6.QtCore import pyqtSignal
                        if hasattr(self.parent(), 'on_ocr_region_selected'):
                            self.parent().on_ocr_region_selected(orig_x, orig_y, orig_w, orig_h)
                
                # Reset OCR drag state
                self._ocr_drag_start = None
                self._ocr_drag_rect = None
                self.updateDisplay()
                return
            
            # Xử lý normal resize mode
            if self._resize_mode != self.RESIZE_NONE:
                self._resize_mode = self.RESIZE_NONE
                self._drag_start_pos = None
                self._rect_original_geometry = None
                self.setCursor(Qt.ArrowCursor)

class ScaledLabel(QLabel):
    """Custom QLabel tự động scale pixmap theo width container"""
    
    def __init__(self):
        super().__init__()
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: var(--card-background);
                padding: 8px;
            }
        """)
    
    def setPixmap(self, pixmap: QPixmap):
        """Override setPixmap để lưu original pixmap"""
        if pixmap:
            self._pixmap = pixmap
            self.updatePixmap()
        else:
            self._pixmap = None
            super().setPixmap(QPixmap())
    
    def resizeEvent(self, event):
        """Auto resize pixmap khi label resize"""
        super().resizeEvent(event)
        if self._pixmap:
            self.updatePixmap()
    
    def updatePixmap(self):
        """Scale pixmap to fit width while keeping aspect ratio"""
        if not self._pixmap or self._pixmap.isNull():
            return
        
        # Scale to fit width of label, keep aspect ratio
        available_width = self.width() - 16  # Trừ padding
        available_height = self.height() - 16
        
        if available_width <= 0 or available_height <= 0:
            return
        
        scaled_pixmap = self._pixmap.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        super().setPixmap(scaled_pixmap)

class CanvasPanel(QWidget):
    """Canvas panel với 2 sub-panels để hiển thị ảnh gốc và kết quả"""
    
    # Signals
    folder_selected = Signal(str)
    image_changed = Signal(int)
    segmentation_completed = Signal(int, QImage)
    text_detection_completed = Signal(int, QImage)
    ocr_completed = Signal(int, list)
    panel_visibility_changed = Signal(bool, bool)
    ocr_region_selected = Signal(int, int, int, int, int)  # x, y, w, h, image_index
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.segmentation_results: dict = {}  # {index: QImage}
        self.visualization_results: dict = {}  # THÊM: {index: QImage} - Step 2 visualization
        self.text_detection_results: dict = {}  # {index: QImage}
        self.ocr_results: dict = {}  # {index: list}
        
        self.image_loader = ImageLoader()
        self.segmentation_processor: Optional[SegmentationProcessor] = None
        self.manga_pipeline_processor: Optional[MangaPipelineProcessor] = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Toggle buttons bar
        toggle_bar = QWidget()
        toggle_bar.setStyleSheet("""
            QWidget {
                background-color: var(--sidebar-background);
                border-bottom: 1px solid var(--border);
            }
        """)
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(8, 4, 8, 4)
        toggle_layout.setSpacing(8)
        
        self.toggle_left_btn = QPushButton("◀ Ẩn gốc")
        self.toggle_left_btn.setCheckable(True)
        self.toggle_left_btn.setStyleSheet(self._get_toggle_button_style())
        self.toggle_left_btn.clicked.connect(self.toggle_left_panel)
        toggle_layout.addWidget(self.toggle_left_btn)
        
        toggle_layout.addStretch()
        
        self.toggle_right_btn = QPushButton("Ẩn kết quả ▶")
        self.toggle_right_btn.setCheckable(True)
        self.toggle_right_btn.setStyleSheet(self._get_toggle_button_style())
        self.toggle_right_btn.clicked.connect(self.toggle_right_panel)
        toggle_layout.addWidget(self.toggle_right_btn)
        
        # Navigation bar với các button icon
        nav_bar = QWidget()
        nav_bar.setStyleSheet("""
            QWidget {
                background-color: var(--sidebar-background);
                border-bottom: 1px solid var(--border);
            }
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 4, 8, 4)
        nav_layout.setSpacing(8)
        
        # Tạo các button với icon (tạm thời chưa có chức năng)
        self.nav_buttons = []
        button_configs = [
            ("🖼️", "Open Folder"),
            ("▶️", "Run Pipeline"),
            ("💾", "Save"),
            ("↺", "Undo"),
            ("↻", "Redo"),
            ("🔍", "Zoom In"),
            ("🔎", "Zoom Out"),
            ("⚙️", "Settings"),
        ]
        
        for icon, tooltip in button_configs:
            btn = QPushButton(icon)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 32)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: var(--card-background);
                    color: var(--text-primary);
                    border: 1px solid var(--border);
                    border-radius: 4px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: var(--sidebar-background);
                    border-color: var(--primary);
                }
                QPushButton:pressed {
                    background-color: var(--primary);
                }
            """)
            btn.setEnabled(False)  # Tạm thời disable vì chưa có chức năng
            self.nav_buttons.append(btn)
            nav_layout.addWidget(btn)
        
        nav_layout.addStretch()
        
        main_layout.addWidget(nav_bar)
        
        # Splitter chứa 2 panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: var(--border);
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: var(--primary);
            }
        """)
        
        # Left panel - Original image
        self.left_panel = ImageDisplayWidget("Original Image")
        self.splitter.addWidget(self.left_panel)
        
        # Right panel
        self.right_panel = ImageDisplayWidget("Output Image")
        self.splitter.addWidget(self.right_panel)
        
        # Set initial sizes (50-50)
        self.splitter.setSizes([1, 1])
        
        main_layout.addWidget(self.splitter, 1)
        
        # Apply focus to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
    
    def setup_connections(self):
        """Setup signal connections"""
        self.image_loader.images_loaded.connect(self.on_images_loaded)
        self.image_loader.error_occurred.connect(self.on_load_error)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key.Key_Up:
            self.previous_image()
            event.accept()
        elif event.key() == Qt.Key.Key_Down:
            self.next_image()
            event.accept()
        else:
            super().keyPressEvent(event)
            
    def mousePressEvent(self, event):
        """Giữ focus khi click vào panel"""
        super().mousePressEvent(event)
        self.setFocus()
        
    def showEvent(self, event):
        """Set focus khi widget được hiển thị"""
        super().showEvent(event)
        self.setFocus()
    
    def open_folder_dialog(self):
        """Mở dialog chọn folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục chứa ảnh",
            os.path.expanduser("~")
        )
        
        if folder:
            self.load_folder(folder)
    
    def load_folder(self, folder_path: str):
        """Load folder với validation"""
        valid_extensions = VALID_IMAGE_EXTENSIONS
        all_files = []
        invalid_files = []
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(valid_extensions):
                    all_files.append(file_path)
                else:
                    invalid_files.append(file)
        
        # Show warning if there are invalid files
        if invalid_files:
            self.show_invalid_files_warning(invalid_files, all_files, folder_path)
        else:
            self.process_valid_files(all_files, folder_path)
    
    def show_invalid_files_warning(self, invalid_files: List[str], 
                                   valid_files: List[str], folder_path: str):
        """Hiển thị cảnh báo file không hợp lệ"""
        # Create custom modal
        modal = CustomModal(
            title="File không hợp lệ",
            size="md",
            parent=self
        )
        
        # Create warning content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        
        warning_text = QLabel(
            f"Tìm thấy {len(invalid_files)} file không hợp lệ.\n"
            f"Chỉ chấp nhận: .jpg, .jpeg, .png"
        )
        warning_text.setStyleSheet("color: var(--text-primary); font-size: 14px;")
        warning_text.setWordWrap(True)
        
        # List some invalid files
        if len(invalid_files) <= 5:
            files_text = "\n".join(invalid_files)
        else:
            files_text = "\n".join(invalid_files[:5]) + f"\n... và {len(invalid_files) - 5} file khác"
        
        files_label = QLabel(files_text)
        files_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            padding: 8px;
            background-color: var(--sidebar-background);
            border-radius: 4px;
        """)
        files_label.setWordWrap(True)
        
        content_layout.addWidget(warning_text)
        content_layout.addWidget(files_label)
        
        modal.setBody(content)
        modal.setFooterVisible(True)
        modal.setActionButton("Continue", "primary")
        modal.cancel_button.setText("Cancel")
        
        # Handle response
        result = modal.exec()
        
        if result == CustomModal.Accepted:
            self.process_valid_files(valid_files, folder_path)
    
    def process_valid_files(self, valid_files: List[str], folder_path: str):
        """Xử lý các file hợp lệ"""
        if not valid_files:
            QMessageBox.warning(
                self,
                "Không có ảnh",
                "Không tìm thấy file ảnh hợp lệ trong thư mục."
            )
            return
                
        # Sort files
        valid_files.sort()
        
        # Load images
        self.image_loader.load_images(valid_files)
        
        # Emit folder selected
        self.folder_selected.emit(folder_path)
    
    def on_images_loaded(self, image_paths: List[str]):
        """Handle images loaded"""
        self.image_paths = image_paths
        self.current_index = 0
        self.segmentation_results.clear()
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        # Display first image
        if self.image_paths:
            self.display_current_image()
        
        # Set focus để nhận keyboard events
        self.setFocus()
    
    def on_load_error(self, error_msg: str):
        """Handle load error"""
        self.logger.error(f"[ERROR] Load error: {error_msg}")
        QMessageBox.critical(self, "Lỗi", error_msg)
    
    def display_current_image(self):
        """Hiển thị ảnh hiện tại với khả năng hiển thị kết quả pipeline"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            self.logger.warning(f"[DISPLAY] Cannot display - no images or invalid index: {self.current_index}")
            return
        
        current_path = self.image_paths[self.current_index]
        
        # Display original image on left panel
        self.left_panel.set_image(image_path=current_path)
        
        # Log available results
        self.logger.info(f"[DISPLAY] Current index: {self.current_index}")
        self.logger.info(f"[DISPLAY] Available results:")
        self.logger.info(f"  - Final results: {list(self.segmentation_results.keys())}")
        self.logger.info(f"  - Visualizations: {list(self.visualization_results.keys())}")
        self.logger.info(f"  - Text detections: {list(self.text_detection_results.keys())}")
        
        # Priority: Final Result (Step 5) > Visualization (Step 2) > Text Detection > Nothing
        if self.current_index in self.segmentation_results:
            self.logger.info(f"[DISPLAY] Showing FINAL RESULT for index {self.current_index}")
            result_data = self.segmentation_results[self.current_index]
            pixmap = QPixmap.fromImage(result_data['image'])
            rectangles = result_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.visualization_results:
            # Hiển thị visualization với hình chữ nhật đỏ (Step 2)
            self.logger.info(f"[DISPLAY] Showing VISUALIZATION for index {self.current_index}")
            vis_data = self.visualization_results[self.current_index]
            pixmap = QPixmap.fromImage(vis_data['image'])
            rectangles = vis_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.text_detection_results:
            self.logger.info(f"[DISPLAY] Showing TEXT DETECTION for index {self.current_index}")
            result_qimage = self.text_detection_results[self.current_index]
            pixmap = QPixmap.fromImage(result_qimage)
            self.right_panel.set_image(pixmap=pixmap)
        else:
            self.logger.warning(f"[DISPLAY] Right panel: NO RESULTS available for index {self.current_index}")
            self.right_panel.clear()
        
        # Emit signal for control panel
        self.image_changed.emit(self.current_index)
    
    def previous_image(self):
        """Chuyển về ảnh trước"""
        if self.image_paths and self.current_index > 0:
            old_index = self.current_index
            self.current_index -= 1
            self.display_current_image()
        else:
            self.logger.warning(f"[NAVIGATE] Cannot go previous - at first image or no images")
    
    def next_image(self):
        """Chuyển sang ảnh sau"""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            old_index = self.current_index
            self.current_index += 1
            self.display_current_image()
        else:
            self.logger.warning(f"[NAVIGATE] Cannot go next - at last image or no images")
    
    def start_segmentation(self):
        """Bắt đầu full manga pipeline (5 steps)"""
        if not self.image_paths:
            self.logger.warning("[PIPELINE] Cannot start - no images loaded")
            QMessageBox.warning(self, "Cảnh báo", "Chưa load ảnh nào.")
            return
        
        # Initialize manga pipeline processor
        if not self.manga_pipeline_processor:
            self.manga_pipeline_processor = MangaPipelineProcessor()
            
            # Connect signals
            self.manga_pipeline_processor.result_ready.connect(self.on_final_result)
            self.manga_pipeline_processor.visualization_ready.connect(self.on_visualization_result)  # THÊM
            self.manga_pipeline_processor.ocr_result_ready.connect(self.on_ocr_result)
            self.manga_pipeline_processor.progress_updated.connect(self.on_pipeline_progress)
            self.manga_pipeline_processor.error_occurred.connect(self.on_pipeline_error)
            self.manga_pipeline_processor.completed.connect(self.on_pipeline_completed)
        
        # Clear previous results
        self.segmentation_results.clear()
        self.visualization_results.clear()  # THÊM
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        # Start processing
        self.manga_pipeline_processor.process_images(self.image_paths)
    
    def on_segmentation_result(self, index: int, result_image: QImage):
        """Handle segmentation result - KHÔNG DÙNG NỮA (chỉ dùng on_final_result)"""
        self.logger.warning(f"[RESULT] on_segmentation_result called (deprecated) - index: {index}")
        self.segmentation_results[index] = result_image
        
        # Update display if this is the current image
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.segmentation_completed.emit(index, result_image)
    
    def on_text_detection_result(self, index: int, result_image: QImage):
        """Handle text detection result - KHÔNG DÙNG NỮA"""
        self.logger.warning(f"[RESULT] on_text_detection_result called (deprecated) - index: {index}")
        self.text_detection_results[index] = result_image
        
        # Update display if this is the current image
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.text_detection_completed.emit(index, result_image)
            
    def on_ocr_result(self, index: int, texts: list):
        """Handle OCR result from original segments"""
        
        # Store result
        self.ocr_results[index] = texts
        
        # Emit signal để ControlPanel có thể hiển thị kết quả OCR
        self.ocr_completed.emit(index, texts)
            
    def on_pipeline_completed(self):
        """Handle pipeline completion"""
        QMessageBox.information(
            self, 
            "Hoàn thành", 
            f"Đã xử lý xong {len(self.image_paths)} ảnh!\n\n"
            "Pipeline gồm 5 bước:\n"
            "1. Segmentation (YOLOv8)\n"
            "2. Tạo blank canvas\n"
            "3. Text detection & removal\n"
            "4. Cập nhật segments\n"
            "5. Paste lại ảnh gốc (Final Result)"
        )
    
    def on_segmentation_error(self, error_msg: str):
        """Handle segmentation error"""
        self.logger.error(f"[ERROR] Segmentation error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Segmentation", error_msg)
        
    def on_final_result(self, index: int, result_image: QImage, rectangles: list = None):
        """Handle final result từ pipeline (Step 5 - Final composition)"""
        
        self.logger.info(f"[FINAL] Received final result for index {index}")
        
        # Rectangles đã được sắp xếp trong pipeline theo thứ tự đọc manga
        # Lưu vào segmentation_results với rectangles metadata
        self.segmentation_results[index] = {
            'image': result_image,
            'rectangles': rectangles if rectangles else []
        }
        self.logger.info(f"[FINAL] Stored final result. Total stored: {len(self.segmentation_results)}")
        self.logger.info(f"[FINAL] Rectangles order (right-to-left, top-to-bottom): {[r['id'] for r in (rectangles if rectangles else [])]}")
        
        # Update display nếu đây là ảnh hiện tại
        if index == self.current_index:
            self.logger.info(f"[FINAL] Displaying final result for current image {index}")
            self.logger.info(f"[FINAL] This will override visualization if it was displayed")
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
        else:
            self.logger.info(f"[FINAL] Not current image (current={self.current_index}, received={index})")
        
        # Emit signal
        self.segmentation_completed.emit(index, result_image)
        
    def on_visualization_result(self, index: int, vis_image: QImage, rectangles: list = None):
        """Handle visualization result (step 2 - với hình chữ nhật đỏ)"""
        
        self.logger.info(f"[VIS] Received visualization for index {index}")
        
        # Rectangles đã được sắp xếp trong pipeline theo thứ tự đọc manga
        # Lưu vào dictionary với rectangles metadata
        self.visualization_results[index] = {
            'image': vis_image,
            'rectangles': rectangles if rectangles else []
        }
        self.logger.info(f"[VIS] Stored visualization. Total stored: {len(self.visualization_results)}")
        self.logger.info(f"[VIS] Rectangles order (right-to-left, top-to-bottom): {[r['id'] for r in (rectangles if rectangles else [])]}")
        
        # Update display nếu đây là ảnh hiện tại
        if index == self.current_index:
            self.logger.info(f"[VIS] Displaying visualization for current image {index}")
            pixmap = QPixmap.fromImage(vis_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
        else:
            self.logger.info(f"[VIS] Not current image (current={self.current_index}, received={index})")
            
    def on_pipeline_progress(self, current: int, total: int, step: str):
        """Handle pipeline progress updates"""
        
        # Cập nhật status text nếu có status bar
        status_text = f"Processing {current}/{total}: {step}"
    
    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline error"""
        self.logger.error(f"Pipeline error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Pipeline", error_msg)
        
    def on_ocr_region_selected(self, x: int, y: int, w: int, h: int):
        """Xử lý khi user chọn vùng OCR"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
        
        # Emit signal với image index hiện tại
        self.ocr_region_selected.emit(x, y, w, h, self.current_index)
        
        self.logger.info(f"[OCR] Region selected: x={x}, y={y}, w={w}, h={h} on image {self.current_index}")
        
        # Tắt OCR mode sau khi chọn xong
        self.left_panel.disable_ocr_mode()
    
    def enable_ocr_selection_mode(self):
        """Bật chế độ chọn vùng OCR"""
        self.left_panel.enable_ocr_mode()
        self.logger.info("[OCR] OCR selection mode enabled")
    
    def disable_ocr_selection_mode(self):
        """Tắt chế độ chọn vùng OCR"""
        self.left_panel.disable_ocr_mode()
        self.logger.info("[OCR] OCR selection mode disabled")
    
    def toggle_left_panel(self):
        """Toggle hiển thị left panel"""
        if self.toggle_left_btn.isChecked():
            self.left_panel.hide()
            self.toggle_left_btn.setText("▶ Hiện gốc")
        else:
            self.left_panel.show()
            self.toggle_left_btn.setText("◀ Ẩn gốc")
        
        # Emit signal để HomePage điều chỉnh layout
        self.panel_visibility_changed.emit(
            self.left_panel.isVisible(),
            self.right_panel.isVisible()
        )
    
    def toggle_right_panel(self):
        """Toggle hiển thị right panel"""
        if self.toggle_right_btn.isChecked():
            self.right_panel.hide()
            self.toggle_right_btn.setText("◀ Hiện kết quả")
        else:
            self.right_panel.show()
            self.toggle_right_btn.setText("Ẩn kết quả ▶")
        
        # Emit signal để HomePage điều chỉnh layout
        self.panel_visibility_changed.emit(
            self.left_panel.isVisible(),
            self.right_panel.isVisible()
        )
    
    def _get_toggle_button_style(self) -> str:
        """Style cho toggle buttons"""
        return """
            QPushButton {
                background-color: var(--card-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: var(--sidebar-background);
                border-color: var(--primary);
            }
            QPushButton:checked {
                background-color: var(--primary);
                color: white;
                border-color: var(--primary);
            }
        """