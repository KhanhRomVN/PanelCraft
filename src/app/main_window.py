from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QStatusBar, QTabWidget)
from PySide6.QtCore import Qt
import logging

from ui.components.sidebar import Sidebar
from ui.components.header import Header
from widgets.home_widget import HomeWidget
from widgets.data_widget import DataWidget
from widgets.settings_widget import SettingsWidget

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
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar, 1)
        
        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Header
        self.header = Header()
        content_layout.addWidget(self.header)
        
        # Tab widget for main content
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(HomeWidget(), "Home")
        self.tab_widget.addTab(DataWidget(), "Data")
        self.tab_widget.addTab(SettingsWidget(self.config, self.theme_manager), "Settings")
        
        content_layout.addWidget(self.tab_widget)
        main_layout.addWidget(content_widget, 4)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_connections(self):
        """Setup signal connections"""
        self.sidebar.tab_changed.connect(self.tab_widget.setCurrentIndex)
        self.header.settings_clicked.connect(self.show_settings_tab)
        self.header.help_clicked.connect(self.show_help)
        
        # Connect theme changes to refresh UI
        self.theme_manager.theme_changed.connect(self.refresh_ui)
        
    def refresh_ui(self):
        """Refresh UI when theme changes"""
        # Re-apply styles or refresh widgets if needed
        pass
        
    def show_settings_tab(self):
        """Show settings tab when settings button is clicked"""
        self.tab_widget.setCurrentIndex(2)  # Settings tab index

    def show_help(self):
        """Show help dialog"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Help", 
                          "PanelCraft Application\n\n"
                          "This is a modern PySide6 application with theme support.")