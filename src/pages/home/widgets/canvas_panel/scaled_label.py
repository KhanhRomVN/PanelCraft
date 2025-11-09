from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from typing import Optional


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
        
        super().setPixmap(scaled_pixmap)