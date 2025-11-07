from PySide6.QtWidgets import (QWidget, QHBoxLayout, QLabel, 
                              QPushButton, QSizePolicy)
from PySide6.QtCore import Qt, Signal

class Header(QWidget):
    """Header component for the main window"""
    
    settings_clicked = Signal()
    help_clicked = Signal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # App title
        title_label = QLabel("PanelCraft")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
        """)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        # Buttons
        self.help_btn = QPushButton("Help")
        self.settings_btn = QPushButton("Settings")
        
        # Style buttons
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """
        self.help_btn.setStyleSheet(button_style)
        self.settings_btn.setStyleSheet(button_style)
        
        # Add widgets to layout
        layout.addWidget(title_label)
        layout.addWidget(spacer)
        layout.addWidget(self.help_btn)
        layout.addWidget(self.settings_btn)
        
        # Connect signals
        self.settings_btn.clicked.connect(self.settings_clicked)
        self.help_btn.clicked.connect(self.help_clicked)