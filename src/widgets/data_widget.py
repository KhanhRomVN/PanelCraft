from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                              QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt

class DataWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Data Management")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 15px;
                color: #333;
            }
        """)
        layout.addWidget(title_label)
        
        # Sample table
        self.table = QTableWidget(5, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Value", "Status"])
        
        # Sample data
        sample_data = [
            ["Item 1", "100", "Active"],
            ["Item 2", "200", "Inactive"],
            ["Item 3", "150", "Active"],
            ["Item 4", "300", "Pending"],
            ["Item 5", "250", "Active"]
        ]
        
        for row, data in enumerate(sample_data):
            for col, value in enumerate(data):
                self.table.setItem(row, col, QTableWidgetItem(value))
        
        # Style table
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(self.table)