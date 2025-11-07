from PySide6.QtWidgets import QPushButton, QHBoxLayout, QWidget, QSizePolicy
from PySide6.QtCore import Qt, Signal, Property, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter
from typing import Optional, Union

class CustomButton(QPushButton):
    """Custom button component with multiple variants and states"""
    
    # Signals
    clicked = Signal()
    
    def __init__(self, 
                 text: str = "",
                 variant: str = "primary",
                 size: str = "md",
                 align: str = "center",
                 loading: bool = False,
                 icon: Optional[QIcon] = None,
                 emoji: str = "",
                 parent=None):
        super().__init__(parent)
        
        self._variant = variant
        self._size = size
        self._align = align
        self._loading = loading
        self._icon = icon
        self._emoji = emoji
        self._text = text
        
        self.setup_ui()
        self.apply_styles()
        
    def setup_ui(self):
        """Setup button layout and properties"""
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Create layout for content
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4 if self._text else 0)
        
        # Set alignment
        alignment_map = {
            "left": Qt.AlignLeft,
            "center": Qt.AlignCenter,
            "right": Qt.AlignRight
        }
        layout.setAlignment(alignment_map.get(self._align, Qt.AlignCenter))
        
    def apply_styles(self):
        """Apply styles based on variant and size"""
        # Size styles
        size_styles = {
            "xs": {
                "padding": "8px 12px",
                "height": "32px",
                "font_size": "12px",
                "icon_size": 12
            },
            "sm": {
                "padding": "10px 16px",
                "height": "38px",
                "font_size": "14px",
                "icon_size": 14
            },
            "md": {
                "padding": "12px 20px",
                "height": "48px",
                "font_size": "16px",
                "icon_size": 16
            },
            "lg": {
                "padding": "16px 24px",
                "height": "56px",
                "font_size": "18px",
                "icon_size": 18
            }
        }
        
        size_config = size_styles.get(self._size, size_styles["md"])
        
        # Variant styles
        variant_styles = {
            "primary": """
                QPushButton {
                    background-color: var(--button-bg);
                    color: var(--button-text);
                    border: 1px solid var(--button-border);
                    border-radius: 6px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: var(--button-bg-hover);
                    border-color: var(--button-border-hover);
                }
                QPushButton:pressed {
                    background-color: var(--button-bg);
                }
                QPushButton:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: var(--button-second-bg);
                    color: var(--text-primary);
                    border: 1px solid var(--border);
                    border-radius: 6px;
                    font-weight: normal;
                }
                QPushButton:hover {
                    background-color: var(--button-second-bg-hover);
                    border-color: var(--border-hover);
                }
                QPushButton:pressed {
                    background-color: var(--button-second-bg);
                }
            """,
            "ghost": """
                QPushButton {
                    background-color: transparent;
                    color: var(--text-primary);
                    border: none;
                    border-radius: 6px;
                    font-weight: normal;
                }
                QPushButton:hover {
                    background-color: var(--sidebar-item-hover);
                }
                QPushButton:pressed {
                    background-color: var(--button-bg);
                    color: var(--button-text);
                }
            """,
            "error": """
                QPushButton {
                    background-color: #dc2626;
                    color: white;
                    border: 1px solid #dc2626;
                    border-radius: 6px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #b91c1c;
                    border-color: #b91c1c;
                }
            """,
            "loading": """
                QPushButton {
                    background-color: #9ca3af;
                    color: white;
                    border: 1px solid #9ca3af;
                    border-radius: 6px;
                    font-weight: 500;
                    cursor: not-allowed;
                }
            """
        }
        
        base_style = f"""
            QPushButton {{
                padding: {size_config['padding']};
                min-height: {size_config['height']};
                font-size: {size_config['font_size']};
            }}
        """
        
        variant_style = variant_styles.get(
            "loading" if self._loading else self._variant, 
            variant_styles["primary"]
        )
        
        self.setStyleSheet(base_style + variant_style)
        
        # Set icon if provided
        if self._icon:
            icon_size = size_config["icon_size"]
            self.setIcon(self._icon)
            self.setIconSize(QSize(icon_size, icon_size))
        
        # Set text
        if self._emoji and not self._text:
            self.setText(self._emoji)
        elif self._text:
            if self._emoji:
                self.setText(f"{self._emoji} {self._text}")
            else:
                self.setText(self._text)
    
    # Properties
    @Property(str)
    def variant(self):
        return self._variant
    
    @variant.setter
    def variant(self, value):
        self._variant = value
        self.apply_styles()
    
    @Property(str)
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
        self.apply_styles()
    
    @Property(bool)
    def loading(self):
        return self._loading
    
    @loading.setter
    def loading(self, value):
        self._loading = value
        self.apply_styles()