from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from typing import List, Optional
import os
from .interactive_image_label import InteractiveImageLabel


class ImageDisplayWidget(QWidget):
    """Widget hiển thị ảnh với scroll và rectangles có thể drag"""
    
    def __init__(self, title: str = ""):
        super().__init__()
        self.current_pixmap: Optional[QPixmap] = None
        self.rectangles: List[dict] = []
        self.ocr_mode_enabled: bool = False
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
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
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid var(--border);
                border-radius: 4px;
                background-color: var(--card-background);
            }
        """)
        
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
    
    def enable_ocr_mode(self):
        """Bật chế độ OCR drag"""
        self.ocr_mode_enabled = True
        self.image_label.enable_ocr_mode()
    
    def disable_ocr_mode(self):
        """Tắt chế độ OCR drag"""
        self.ocr_mode_enabled = False
        self.image_label.disable_ocr_mode()