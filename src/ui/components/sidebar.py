from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, 
                              QButtonGroup, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Signal, Qt

class Sidebar(QWidget):
    tab_changed = Signal(int)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        # Button group for tabs
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        
        # Tab buttons
        tabs = [
            ("Home", "🏠"),
            ("Data", "📊"),
            ("Settings", "⚙️")
        ]
        
        for i, (text, icon) in enumerate(tabs):
            btn = QPushButton(f"{icon} {text}")
            btn.setCheckable(True)
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 10px;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:checked {
                    background-color: #007acc;
                    color: white;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
            """)
            self.button_group.addButton(btn, i)
            layout.addWidget(btn)
            
        # Spacer
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Connect signals
        self.button_group.idClicked.connect(self.tab_changed.emit)
        
        # Select first tab by default
        if self.button_group.buttons():
            self.button_group.buttons()[0].setChecked(True)