from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QStatusBar, QTabWidget)
from PySide6.QtCore import Qt
import logging
import os
import sys
from pages.home.home_page import HomePage


class MainWindow(QMainWindow):
    def __init__(self, config, theme_manager):
        super().__init__()
        self.config = config
        self.theme_manager = theme_manager
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup main window UI"""
        self.setWindowTitle("PanelCraft")
        # Không set geometry cố định, sẽ dùng showMaximized()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Content area - Direct HomePage without TabWidget
        self.home_widget = HomePage()
        main_layout.addWidget(self.home_widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_connections(self):
        """Setup signal connections"""
        # Connect theme changes to refresh UI
        self.theme_manager.theme_changed.connect(self.refresh_ui)
        
    def refresh_ui(self):
        """Refresh UI when theme changes"""
        # Re-apply styles or refresh widgets if needed
        pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        event.accept()