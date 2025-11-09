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
            
            # Kiểm tra và tải models nếu cần
            from core.model_manager import ModelManager
            from widget.modal.model_download_modal import ModelDownloadModal
            
            self.model_manager = ModelManager()
            
            # Kiểm tra xem có cần tải models không
            if not self.model_manager.is_setup_complete():
                
                # Hiển thị modal tải models
                modal = ModelDownloadModal(self.model_manager)
                result = modal.exec()
                
                if result != ModelDownloadModal.Accepted:
                    self.logger.warning("User cancelled model download")
                    return False
            
            # Create main window
            self.main_window = MainWindow(self.config, self.theme_manager)
            self.main_window.showMaximized()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            return False