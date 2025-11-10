from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextOption

class AutoResizeTextEdit(QTextEdit):
    """QTextEdit tự động điều chỉnh height theo nội dung (tối đa max_lines dòng)"""
    
    textChanged = Signal()
    
    def __init__(self, 
                 placeholder: str = "",
                 min_lines: int = 1,
                 max_lines: int = 15,
                 parent=None):
        super().__init__(parent)
        
        self._min_lines = min_lines
        self._max_lines = max_lines
        
        self.setPlaceholderText(placeholder)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        
        # Connect textChanged
        self.document().contentsChanged.connect(self._on_text_changed)
        
        # Set initial height
        self._update_height()
    
    def _on_text_changed(self):
        """Handle text change và emit signal"""
        self._update_height()
        self.textChanged.emit()
    
    def _update_height(self):
        """Cập nhật height dựa trên số dòng thực tế"""
        doc = self.document()
        doc.setTextWidth(self.viewport().width())
        
        # Tính số dòng
        doc_height = doc.size().height()
        line_height = self.fontMetrics().lineSpacing()
        num_lines = int(doc_height / line_height) + 1
        
        # Clamp số dòng trong khoảng [min_lines, max_lines]
        num_lines = max(self._min_lines, min(num_lines, self._max_lines))
        
        # Tính height mới
        padding = 16  # Top + bottom padding
        new_height = num_lines * line_height + padding
        
        self.setMinimumHeight(new_height)
        self.setMaximumHeight(new_height)