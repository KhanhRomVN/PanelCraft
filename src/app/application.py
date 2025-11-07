from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon
import logging

from .main_window import MainWindow
from core.config import AppConfig
from core.theme import ThemeManager

class Application(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("PanelCraft")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("PanelCraft")
        
        self.config = AppConfig()
        self.theme_manager = ThemeManager()
        self.main_window = None
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """Khởi tạo ứng dụng"""
        try:
            # Load configuration
            self.config.load()
            
            # Setup theme (this will apply the stylesheet)
            # No need to setup separate stylesheet as theme manager handles it
            
            # Create main window
            self.main_window = MainWindow(self.config, self.theme_manager)
            self.main_window.show()
            
            self.logger.info("Application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            return False