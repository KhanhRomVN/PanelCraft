from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QFont, QFontMetrics
from typing import List, Optional
import logging
from ...constants.constants import RESIZE_HANDLE_SIZE, MIN_RECTANGLE_SIZE
from ...types.home_types import RectangleDict


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
        self._handle_size = RESIZE_HANDLE_SIZE
        
        self._ocr_mode: bool = False
        self._ocr_drag_start = None
        self._ocr_drag_rect = None
        
        self.logger = logging.getLogger(__name__)
        
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
        self._ocr_mode = True
        self.setCursor(Qt.CrossCursor)
    
    def disable_ocr_mode(self):
        """Tắt chế độ OCR drag-and-drop"""
        self._ocr_mode = False
        self._ocr_drag_start = None
        self._ocr_drag_rect = None
        self.setCursor(Qt.ArrowCursor)
        self.updateDisplay()
    
    def resizeEvent(self, event):
        """Auto resize khi label resize"""
        super().resizeEvent(event)
        if self._pixmap:
            self.updateDisplay()
    
    def updateDisplay(self):
        """Vẽ pixmap + rectangles"""
        if not self._pixmap or self._pixmap.isNull():
            return
        
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
        
        if self._rectangles or self._ocr_drag_rect:
            display_pixmap = self._draw_overlays(scaled_pixmap)
            super().setPixmap(display_pixmap)
        else:
            super().setPixmap(scaled_pixmap)
    
    def _draw_overlays(self, scaled_pixmap: QPixmap) -> QPixmap:
        """Vẽ rectangles và OCR drag rectangle lên pixmap"""
        from PySide6.QtCore import QRect
        
        display_pixmap = scaled_pixmap.copy()
        painter = QPainter(display_pixmap)
        
        if self._rectangles:
            self._draw_rectangles(painter, scaled_pixmap)
        
        if self._ocr_drag_rect:
            self._draw_ocr_drag(painter)
        
        painter.end()
        return display_pixmap
    
    def _draw_rectangles(self, painter: QPainter, scaled_pixmap: QPixmap):
        """Vẽ rectangles với numbers và handles"""
        from PySide6.QtCore import QRect
        
        scale_x = scaled_pixmap.width() / self._pixmap.width()
        scale_y = scaled_pixmap.height() / self._pixmap.height()
        
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        
        for idx, rect in enumerate(self._rectangles):
            x = int(rect['x'] * scale_x)
            y = int(rect['y'] * scale_y)
            w = int(rect['w'] * scale_x)
            h = int(rect['h'] * scale_y)
            
            painter.drawRect(QRect(x, y, w, h))
            
            self._draw_rectangle_number(painter, idx + 1, x, y, w, h)
            
            if rect['id'] == self._active_rect_id:
                self._draw_resize_handles(painter, x, y, w, h)
        
        painter.setBrush(Qt.NoBrush)
    
    def _draw_rectangle_number(self, painter: QPainter, number: int, x: int, y: int, w: int, h: int):
        """Vẽ số thứ tự ở góc phải trên"""
        number_text = str(number)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(number_text)
        text_height = fm.height()
        
        number_x = x + w + 5
        number_y = y - 5
        
        painter.setBrush(QBrush(Qt.red))
        painter.setPen(Qt.NoPen)
        padding = 4
        painter.drawRect(
            number_x - padding,
            number_y - text_height - padding,
            text_width + 2 * padding,
            text_height + 2 * padding
        )
        
        painter.setPen(QPen(Qt.white))
        painter.drawText(number_x, number_y - padding, number_text)
        
        painter.setPen(QPen(Qt.red, 2))
        painter.setBrush(Qt.NoBrush)
    
    def _draw_resize_handles(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Vẽ handles cho resize"""
        handle_size = self._handle_size
        painter.setBrush(QBrush(Qt.red))
        
        handles = [
            (x - handle_size//2, y - handle_size//2),
            (x + w - handle_size//2, y - handle_size//2),
            (x + w - handle_size//2, y + h - handle_size//2),
            (x - handle_size//2, y + h - handle_size//2),
            (x + w//2 - handle_size//2, y - handle_size//2),
            (x + w - handle_size//2, y + h//2 - handle_size//2),
            (x + w//2 - handle_size//2, y + h - handle_size//2),
            (x - handle_size//2, y + h//2 - handle_size//2),
        ]
        
        for hx, hy in handles:
            painter.drawRect(hx, hy, handle_size, handle_size)
        
        painter.setBrush(Qt.NoBrush)
    
    def _draw_ocr_drag(self, painter: QPainter):
        """Vẽ OCR drag rectangle"""
        painter.setPen(QPen(Qt.blue, 3, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        
        x, y, w, h = self._ocr_drag_rect
        painter.drawRect(int(x), int(y), int(w), int(h))
    
    def mousePressEvent(self, event):
        """Bắt đầu drag/resize rectangle HOẶC OCR drag"""
        if not self._pixmap or event.button() != Qt.LeftButton:
            return
        
        if self._ocr_mode:
            self._handle_ocr_press(event)
            return
        
        self._handle_rectangle_press(event)
    
    def _handle_ocr_press(self, event):
        """Xử lý OCR mode press"""
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y
        
        if 0 <= rel_x <= pixmap_rect.width() and 0 <= rel_y <= pixmap_rect.height():
            self._ocr_drag_start = (rel_x, rel_y)
            self._ocr_drag_rect = None
    
    def _handle_rectangle_press(self, event):
        """Xử lý rectangle interaction press"""
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y
        
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap_rect.width() or rel_y > pixmap_rect.height():
            return
        
        scale_x = self._pixmap.width() / pixmap_rect.width()
        scale_y = self._pixmap.height() / pixmap_rect.height()
        
        orig_x = int(rel_x * scale_x)
        orig_y = int(rel_y * scale_y)
        
        for rect in reversed(self._rectangles):
            mode = self._get_resize_mode(orig_x, orig_y, rect)
            
            if mode != self.RESIZE_NONE:
                self._active_rect_id = rect['id']
                self._resize_mode = mode
                self._drag_start_pos = (orig_x, orig_y)
                self._rect_original_geometry = (rect['x'], rect['y'], rect['w'], rect['h'])
                
                if mode == self.MOVE:
                    self.setCursor(Qt.ClosedHandCursor)
                else:
                    self.setCursor(self._get_cursor_for_mode(mode))
                
                self.updateDisplay()
                break
    
    def mouseMoveEvent(self, event):
        """Di chuyển/resize rectangle HOẶC vẽ OCR drag rectangle"""
        if not self._pixmap:
            return
        
        if self._ocr_mode and self._ocr_drag_start:
            self._handle_ocr_drag(event)
            return
        
        self._handle_rectangle_move(event)
    
    def _handle_ocr_drag(self, event):
        """Xử lý OCR drag"""
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = max(0, min(click_pos.x() - offset_x, pixmap_rect.width()))
        rel_y = max(0, min(click_pos.y() - offset_y, pixmap_rect.height()))
        
        start_x, start_y = self._ocr_drag_start
        
        x = min(start_x, rel_x)
        y = min(start_y, rel_y)
        w = abs(rel_x - start_x)
        h = abs(rel_y - start_y)
        
        self._ocr_drag_rect = (x, y, w, h)
        self.updateDisplay()
    
    def _handle_rectangle_move(self, event):
        """Xử lý rectangle move/resize"""
        click_pos = event.pos()
        pixmap_rect = self.pixmap().rect()
        
        offset_x = (self.width() - pixmap_rect.width()) // 2
        offset_y = (self.height() - pixmap_rect.height()) // 2
        
        rel_x = click_pos.x() - offset_x
        rel_y = click_pos.y() - offset_y
        
        if rel_x < 0 or rel_y < 0 or rel_x > pixmap_rect.width() or rel_y > pixmap_rect.height():
            if self._resize_mode == self.RESIZE_NONE:
                self.setCursor(Qt.ArrowCursor)
            return
        
        scale_x = self._pixmap.width() / pixmap_rect.width()
        scale_y = self._pixmap.height() / pixmap_rect.height()
        
        orig_x = int(rel_x * scale_x)
        orig_y = int(rel_y * scale_y)
        
        if self._resize_mode == self.RESIZE_NONE:
            self._update_cursor_for_hover(orig_x, orig_y)
            return
        
        self._apply_resize_or_move(orig_x, orig_y)
    
    def _update_cursor_for_hover(self, orig_x: int, orig_y: int):
        """Cập nhật cursor khi hover"""
        for rect in reversed(self._rectangles):
            mode = self._get_resize_mode(orig_x, orig_y, rect)
            if mode != self.RESIZE_NONE:
                self.setCursor(self._get_cursor_for_mode(mode))
                return
        self.setCursor(Qt.ArrowCursor)
    
    def _apply_resize_or_move(self, orig_x: int, orig_y: int):
        """Áp dụng resize hoặc move cho active rectangle"""
        for rect in self._rectangles:
            if rect['id'] != self._active_rect_id:
                continue
            
            dx = orig_x - self._drag_start_pos[0]
            dy = orig_y - self._drag_start_pos[1]
            
            orig_x_rect, orig_y_rect, orig_w_rect, orig_h_rect = self._rect_original_geometry
            min_size = MIN_RECTANGLE_SIZE
            
            if self._resize_mode == self.MOVE:
                self._handle_move(rect, dx, dy, orig_x_rect, orig_y_rect)
            else:
                self._handle_resize(rect, dx, dy, orig_x_rect, orig_y_rect, orig_w_rect, orig_h_rect, min_size)
            
            break
        
        self.updateDisplay()
    
    def _handle_move(self, rect: dict, dx: int, dy: int, orig_x: int, orig_y: int):
        """Xử lý move rectangle"""
        new_x = orig_x + dx
        new_y = orig_y + dy
        rect['x'] = max(0, min(new_x, self._pixmap.width() - rect['w']))
        rect['y'] = max(0, min(new_y, self._pixmap.height() - rect['h']))
    
    def _handle_resize(self, rect: dict, dx: int, dy: int, orig_x: int, orig_y: int, orig_w: int, orig_h: int, min_size: int):
        """Xử lý resize rectangle"""
        mode = self._resize_mode
        
        if mode == self.RESIZE_TOP_LEFT:
            new_x = max(0, min(orig_x + dx, orig_x + orig_w - min_size))
            new_y = max(0, min(orig_y + dy, orig_y + orig_h - min_size))
            rect['x'], rect['y'] = new_x, new_y
            rect['w'] = orig_x + orig_w - new_x
            rect['h'] = orig_y + orig_h - new_y
        
        elif mode == self.RESIZE_TOP:
            new_y = max(0, min(orig_y + dy, orig_y + orig_h - min_size))
            rect['y'] = new_y
            rect['h'] = orig_y + orig_h - new_y
        
        elif mode == self.RESIZE_TOP_RIGHT:
            new_y = max(0, min(orig_y + dy, orig_y + orig_h - min_size))
            new_w = max(min_size, min(orig_w + dx, self._pixmap.width() - orig_x))
            rect['y'] = new_y
            rect['w'] = new_w
            rect['h'] = orig_y + orig_h - new_y
        
        elif mode == self.RESIZE_RIGHT:
            rect['w'] = max(min_size, min(orig_w + dx, self._pixmap.width() - orig_x))
        
        elif mode == self.RESIZE_BOTTOM_RIGHT:
            rect['w'] = max(min_size, min(orig_w + dx, self._pixmap.width() - orig_x))
            rect['h'] = max(min_size, min(orig_h + dy, self._pixmap.height() - orig_y))
        
        elif mode == self.RESIZE_BOTTOM:
            rect['h'] = max(min_size, min(orig_h + dy, self._pixmap.height() - orig_y))
        
        elif mode == self.RESIZE_BOTTOM_LEFT:
            new_x = max(0, min(orig_x + dx, orig_x + orig_w - min_size))
            rect['x'] = new_x
            rect['w'] = orig_x + orig_w - new_x
            rect['h'] = max(min_size, min(orig_h + dy, self._pixmap.height() - orig_y))
        
        elif mode == self.RESIZE_LEFT:
            new_x = max(0, min(orig_x + dx, orig_x + orig_w - min_size))
            rect['x'] = new_x
            rect['w'] = orig_x + orig_w - new_x
    
    def mouseReleaseEvent(self, event):
        """Kết thúc drag/resize HOẶC OCR drag"""
        if event.button() != Qt.LeftButton:
            return
        
        if self._ocr_mode and self._ocr_drag_start and self._ocr_drag_rect:
            self._finish_ocr_drag()
            return
        
        if self._resize_mode != self.RESIZE_NONE:
            self._resize_mode = self.RESIZE_NONE
            self._drag_start_pos = None
            self._rect_original_geometry = None
            self.setCursor(Qt.ArrowCursor)
    
    def _finish_ocr_drag(self):
        """Hoàn thành OCR drag"""
        x, y, w, h = self._ocr_drag_rect
        
        if w >= 20 and h >= 20:
            if self._pixmap:
                pixmap_rect = self.pixmap().rect()
                scale_x = self._pixmap.width() / pixmap_rect.width()
                scale_y = self._pixmap.height() / pixmap_rect.height()
                
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                orig_w = int(w * scale_x)
                orig_h = int(h * scale_y)
                
                if hasattr(self.parent(), 'on_ocr_region_selected'):
                    self.parent().on_ocr_region_selected(orig_x, orig_y, orig_w, orig_h)
        
        self._ocr_drag_start = None
        self._ocr_drag_rect = None
        self.updateDisplay()
    
    def _get_resize_mode(self, orig_x: int, orig_y: int, rect: dict) -> int:
        """Xác định resize mode dựa trên vị trí click"""
        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
        tolerance = self._handle_size
        
        if abs(orig_x - x) <= tolerance and abs(orig_y - y) <= tolerance:
            return self.RESIZE_TOP_LEFT
        if abs(orig_x - (x + w)) <= tolerance and abs(orig_y - y) <= tolerance:
            return self.RESIZE_TOP_RIGHT
        if abs(orig_x - (x + w)) <= tolerance and abs(orig_y - (y + h)) <= tolerance:
            return self.RESIZE_BOTTOM_RIGHT
        if abs(orig_x - x) <= tolerance and abs(orig_y - (y + h)) <= tolerance:
            return self.RESIZE_BOTTOM_LEFT
        
        if abs(orig_y - y) <= tolerance and x <= orig_x <= x + w:
            return self.RESIZE_TOP
        if abs(orig_x - (x + w)) <= tolerance and y <= orig_y <= y + h:
            return self.RESIZE_RIGHT
        if abs(orig_y - (y + h)) <= tolerance and x <= orig_x <= x + w:
            return self.RESIZE_BOTTOM
        if abs(orig_x - x) <= tolerance and y <= orig_y <= y + h:
            return self.RESIZE_LEFT
        
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