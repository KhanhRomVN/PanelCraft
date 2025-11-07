from PySide6.QtWidgets import (QLineEdit, QVBoxLayout, QHBoxLayout, QLabel, 
                              QWidget, QPushButton, QTextEdit, QSizePolicy)
from PySide6.QtCore import Qt, Signal, Property, QTimer
from PySide6.QtGui import QIcon, QPixmap, QPainter
from typing import Optional, Callable

class CustomInput(QWidget):
    """Custom input component with multiple variants and features"""
    
    # Signals
    textChanged = Signal(str)
    returnPressed = Signal()
    editingFinished = Signal()
    
    def __init__(self,
                 label: str = "",
                 placeholder: str = "",
                 variant: str = "default",
                 size: str = "md",
                 required: bool = False,
                 error: str = "",
                 success: str = "",
                 hint: str = "",
                 show_char_count: bool = False,
                 max_length: int = 0,
                 left_icon: Optional[QIcon] = None,
                 right_icon: Optional[QIcon] = None,
                 multiline: bool = False,
                 rows: int = 3,
                 parent=None):
        super().__init__(parent)
        
        self._label = label
        self._placeholder = placeholder
        self._variant = variant
        self._size = size
        self._required = required
        self._error = error
        self._success = success
        self._hint = hint
        self._show_char_count = show_char_count
        self._max_length = max_length
        self._left_icon = left_icon
        self._right_icon = right_icon
        self._multiline = multiline
        self._rows = rows
        
        self.setup_ui()
        self.apply_styles()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup input layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Label
        if self._label:
            self.label_widget = QLabel(self._label)
            if self._required:
                self.label_widget.setText(f"{self._label} *")
            layout.addWidget(self.label_widget)
        
        # Input container
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)
        
        # Left icon
        if self._left_icon:
            self.left_icon_label = QLabel()
            self.left_icon_label.setPixmap(self._left_icon.pixmap(16, 16))
            self.left_icon_label.setStyleSheet("padding: 0 8px;")
            input_layout.addWidget(self.left_icon_label)
        
        # Input field
        if self._multiline:
            self.input_field = QTextEdit()
            self.input_field.setPlaceholderText(self._placeholder)
            self.input_field.setMaximumHeight(self._rows * 24)
        else:
            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText(self._placeholder)
        
        input_layout.addWidget(self.input_field)
        
        # Right icon
        if self._right_icon:
            self.right_icon_label = QLabel()
            self.right_icon_label.setPixmap(self._right_icon.pixmap(16, 16))
            self.right_icon_label.setStyleSheet("padding: 0 8px;")
            input_layout.addWidget(self.right_icon_label)
        
        layout.addWidget(input_container)
        
        # Character count
        if self._show_char_count and self._max_length > 0:
            self.char_count_label = QLabel("0/0")
            self.char_count_label.setAlignment(Qt.AlignRight)
            self.char_count_label.setStyleSheet("font-size: 12px; color: #6b7280;")
            layout.addWidget(self.char_count_label)
        
        # Error/Success/Hint message
        if self._error or self._success or self._hint:
            self.message_label = QLabel(self._error or self._success or self._hint)
            message_style = "font-size: 12px;"
            if self._error:
                message_style += "color: #dc2626;"
            elif self._success:
                message_style += "color: #16a34a;"
            else:
                message_style += "color: #6b7280;"
            self.message_label.setStyleSheet(message_style)
            layout.addWidget(self.message_label)
    
    def apply_styles(self):
        """Apply styles based on variant and size"""
        # Size styles
        size_styles = {
            "sm": {
                "padding": "8px 12px",
                "font_size": "14px",
                "height": "32px"
            },
            "md": {
                "padding": "12px 16px",
                "font_size": "16px",
                "height": "40px"
            },
            "lg": {
                "padding": "16px 20px",
                "font_size": "18px",
                "height": "48px"
            }
        }
        
        size_config = size_styles.get(self._size, size_styles["md"])
        
        # Variant styles
        variant_styles = {
            "default": """
                QLineEdit, QTextEdit {
                    background-color: var(--input-background);
                    border: 1px solid var(--border);
                    border-radius: 6px;
                    color: var(--text-primary);
                }
                QLineEdit:focus, QTextEdit:focus {
                    border-color: var(--primary);
                    outline: none;
                }
            """,
            "filled": """
                QLineEdit, QTextEdit {
                    background-color: var(--sidebar-background);
                    border: 1px solid transparent;
                    border-radius: 6px;
                    color: var(--text-primary);
                }
                QLineEdit:focus, QTextEdit:focus {
                    background-color: var(--input-background);
                    border-color: var(--primary);
                }
            """,
            "outlined": """
                QLineEdit, QTextEdit {
                    background-color: transparent;
                    border: 2px solid var(--border);
                    border-radius: 6px;
                    color: var(--text-primary);
                }
                QLineEdit:focus, QTextEdit:focus {
                    border-color: var(--primary);
                }
            """,
            "underlined": """
                QLineEdit, QTextEdit {
                    background-color: transparent;
                    border: none;
                    border-bottom: 2px solid var(--border);
                    border-radius: 0px;
                    color: var(--text-primary);
                }
                QLineEdit:focus, QTextEdit:focus {
                    border-bottom-color: var(--primary);
                }
            """
        }
        
        base_style = f"""
            QLineEdit, QTextEdit {{
                padding: {size_config['padding']};
                font-size: {size_config['font_size']};
                min-height: {size_config['height']};
            }}
        """
        
        variant_style = variant_styles.get(self._variant, variant_styles["default"])
        
        # Apply error state
        if self._error:
            variant_style = variant_style.replace("var(--border)", "#dc2626")
            variant_style = variant_style.replace("var(--primary)", "#dc2626")
        
        self.input_field.setStyleSheet(base_style + variant_style)
    
    def setup_connections(self):
        """Setup signal connections"""
        if self._multiline:
            self.input_field.textChanged.connect(self.on_text_changed)
        else:
            self.input_field.textChanged.connect(self.on_text_changed)
            self.input_field.returnPressed.connect(self.returnPressed.emit)
            self.input_field.editingFinished.connect(self.editingFinished.emit)
    
    def on_text_changed(self):
        """Handle text changes"""
        text = self.text()
        self.textChanged.emit(text)
        
        # Update character count
        if self._show_char_count and self._max_length > 0:
            if hasattr(self, 'char_count_label'):
                self.char_count_label.setText(f"{len(text)}/{self._max_length}")
                
                # Change color if approaching limit
                if len(text) > self._max_length * 0.9:
                    self.char_count_label.setStyleSheet("font-size: 12px; color: #dc2626;")
                else:
                    self.char_count_label.setStyleSheet("font-size: 12px; color: #6b7280;")
    
    # Properties and methods
    def text(self) -> str:
        if self._multiline:
            return self.input_field.toPlainText()
        return self.input_field.text()
    
    def setText(self, text: str):
        if self._multiline:
            self.input_field.setPlainText(text)
        else:
            self.input_field.setText(text)
    
    def clear(self):
        self.input_field.clear()
    
    @Property(str)
    def placeholder(self):
        return self._placeholder
    
    @placeholder.setter
    def placeholder(self, value):
        self._placeholder = value
        self.input_field.setPlaceholderText(value)
    
    @Property(bool)
    def required(self):
        return self._required
    
    @required.setter
    def required(self, value):
        self._required = value
        if hasattr(self, 'label_widget'):
            self.label_widget.setText(f"{self._label} *" if value else self._label)
    
    @Property(str)
    def error(self):
        return self._error
    
    @error.setter
    def error(self, value):
        self._error = value
        self.apply_styles()
        if hasattr(self, 'message_label'):
            self.message_label.setText(value)
            self.message_label.setStyleSheet("font-size: 12px; color: #dc2626;")