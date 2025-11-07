from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QWidget, QSizePolicy, QFrame)
from PySide6.QtCore import Qt, Signal, Property, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QMouseEvent
from typing import Optional

class CustomModal(QDialog):
    """Custom modal dialog component"""
    
    # Signals
    accepted = Signal()
    rejected = Signal()
    
    def __init__(self,
                 title: str = "",
                 size: str = "md",
                 show_close_button: bool = True,
                 parent=None):
        super().__init__(parent)
        
        self._title = title
        self._size = size
        self._show_close_button = show_close_button
        
        self.setup_ui()
        self.apply_styles()
        self.setup_connections()
        
        # Dialog properties
        self.setModal(True)
        self.setWindowFlags(Qt.Modal | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
    def setup_ui(self):
        """Setup modal layout"""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)
        
        # Background overlay
        self.overlay = QWidget()
        self.overlay.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 12px;
        """)
        
        # Modal content
        self.modal_content = QFrame()
        self.modal_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        content_layout = QVBoxLayout(self.modal_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Header
        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(24, 16, 24, 16)
        header_layout.setSpacing(12)
        
        # Title
        self.title_label = QLabel(self._title)
        self.title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
        """)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Close button
        self.close_button = QPushButton("×")
        self.close_button.setFixedSize(32, 32)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 6px;
                font-size: 20px;
                color: var(--text-secondary);
            }
            QPushButton:hover {
                background-color: var(--sidebar-item-hover);
                color: var(--text-primary);
            }
        """)
        self.close_button.setVisible(self._show_close_button)
        
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(spacer)
        header_layout.addWidget(self.close_button)
        
        # Body container
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(24, 0, 24, 0)
        
        # Footer
        self.footer = QWidget()
        self.footer_layout = QHBoxLayout(self.footer)
        self.footer_layout.setContentsMargins(24, 16, 24, 24)
        self.footer_layout.setSpacing(12)
        
        # Footer spacer
        footer_spacer = QWidget()
        footer_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Action buttons
        self.cancel_button = QPushButton("Cancel")
        self.action_button = QPushButton("Confirm")
        
        self.footer_layout.addWidget(footer_spacer)
        self.footer_layout.addWidget(self.cancel_button)
        self.footer_layout.addWidget(self.action_button)
        
        # Assemble modal
        content_layout.addWidget(self.header)
        content_layout.addWidget(self.body)
        content_layout.addWidget(self.footer)
        
        # Add to main layout
        main_layout.addWidget(self.overlay)
        main_layout.addWidget(self.modal_content)
        
        # Initially hide footer
        self.footer.setVisible(False)
    
    def apply_styles(self):
        """Apply styles based on size"""
        size_styles = {
            "sm": {"width": "400px", "max_width": "90vw"},
            "md": {"width": "500px", "max_width": "90vw"},
            "lg": {"width": "600px", "max_width": "90vw"},
            "xl": {"width": "800px", "max_width": "90vw"}
        }
        
        size_config = size_styles.get(self._size, size_styles["md"])
        
        modal_style = f"""
            QFrame {{
                background-color: var(--modal-background);
                border-radius: 12px;
                border: 1px solid var(--border);
                min-width: {size_config['width']};
                max-width: {size_config['max_width']};
            }}
        """
        
        self.modal_content.setStyleSheet(modal_style)
        
        # Button styles
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
                min-height: 36px;
            }
        """
        
        cancel_style = """
            QPushButton {
                background-color: var(--button-second-bg);
                color: var(--text-primary);
                border: 1px solid var(--border);
            }
            QPushButton:hover {
                background-color: var(--button-second-bg-hover);
            }
        """
        
        action_style = """
            QPushButton {
                background-color: var(--button-bg);
                color: var(--button-text);
                border: 1px solid var(--button-border);
            }
            QPushButton:hover {
                background-color: var(--button-bg-hover);
            }
        """
        
        self.cancel_button.setStyleSheet(button_style + cancel_style)
        self.action_button.setStyleSheet(button_style + action_style)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.close_button.clicked.connect(self.reject)
        self.cancel_button.clicked.connect(self.reject)
        self.action_button.clicked.connect(self.accept)
        
        # Connect signals
        self.accepted.connect(self.on_accepted)
        self.rejected.connect(self.on_rejected)
    
    def on_accepted(self):
        """Handle accept"""
        self.close()
    
    def on_rejected(self):
        """Handle reject"""
        self.close()
    
    def setBody(self, widget: QWidget):
        """Set body content"""
        # Clear existing body content
        for i in reversed(range(self.body_layout.count())):
            item = self.body_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        
        self.body_layout.addWidget(widget)
    
    def setFooterVisible(self, visible: bool):
        """Show/hide footer"""
        self.footer.setVisible(visible)
    
    def setActionButton(self, text: str, variant: str = "primary"):
        """Set action button properties"""
        self.action_button.setText(text)
        
        # Update variant styles if needed
        variant_styles = {
            "primary": """
                background-color: var(--button-bg);
                color: var(--button-text);
            """,
            "danger": """
                background-color: #dc2626;
                color: white;
            """,
            "success": """
                background-color: #16a34a;
                color: white;
            """
        }
        
        style = variant_styles.get(variant, variant_styles["primary"])
        self.action_button.setStyleSheet(f"QPushButton {{ {style} }}")
    
    def showEvent(self, event):
        """Animate show event"""
        super().showEvent(event)
        self.animate_show()
    
    def animate_show(self):
        """Animate modal appearance"""
        animation = QPropertyAnimation(self.modal_content, b"windowOpacity")
        animation.setDuration(200)
        animation.setStartValue(0)
        animation.setEndValue(1)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        animation.start()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Close modal when clicking outside"""
        if event.button() == Qt.LeftButton:
            if not self.modal_content.geometry().contains(event.pos()):
                self.reject()
        super().mousePressEvent(event)