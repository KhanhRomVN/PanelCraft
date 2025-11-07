from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                              QPushButton, QMessageBox)
from PySide6.QtCore import Qt

class HomeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Welcome label
        welcome_label = QLabel("Welcome to PySide6 Application")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        layout.addWidget(welcome_label)
        
        # Sample button
        demo_btn = QPushButton("Click Me!")
        demo_btn.setFixedSize(150, 50)
        demo_btn.clicked.connect(self.show_message)
        layout.addWidget(demo_btn, 0, Qt.AlignCenter)
        
        layout.addStretch()
        
    def show_message(self):
        QMessageBox.information(self, "Demo", "Hello from PySide6!")