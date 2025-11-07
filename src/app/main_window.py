from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QStatusBar, QTabWidget)
from PySide6.QtCore import Qt
import logging
import os
import sys
from pages.home.home_page import HomePage
from utils.hotload_manager import HotloadManager


class MainWindow(QMainWindow):
    def __init__(self, config, theme_manager):
        super().__init__()
        self.config = config
        self.theme_manager = theme_manager
        self.logger = logging.getLogger(__name__)
        
        # Setup hotload manager
        self.hotload_manager = None
        if config.enable_hotload:
            # Lấy đường dẫn tuyệt đối đến thư mục src
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Danh sách thư mục không cần theo dõi
            exclude_dirs = {'__pycache__', 'venv', '.git', '.pytest_cache', 'resources'}
            
            # Tự động lấy tất cả thư mục con trong src
            watch_paths = []
            for item in os.listdir(src_dir):
                item_path = os.path.join(src_dir, item)
                if os.path.isdir(item_path) and item not in exclude_dirs:
                    watch_paths.append(item_path)
            self.hotload_manager = HotloadManager(watch_paths=watch_paths, enabled=True)
            self.hotload_manager.reload_requested.connect(self.on_hotload_requested)
            self.logger.info(f"Hotload enabled, watching: {watch_paths}")
        
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
        
        # Content area - Direct HomePage without TabWidget
        self.home_widget = HomePage()
        main_layout.addWidget(self.home_widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_connections(self):
        """Setup signal connections"""
        # Connect theme changes to refresh UI
        self.theme_manager.theme_changed.connect(self.refresh_ui)
        
    def on_hotload_requested(self, module_path):
        """Handle hotload request"""
        self.logger.info(f"Hotload requested for: {module_path}")
        
        if self.hotload_manager:
            success = self.hotload_manager.reload_module(module_path)
            
            if success:
                # Reload the widget
                if 'home_page' in module_path.lower():
                    self.reload_home_page()
                
                self.statusBar().showMessage(f"Reloaded: {module_path}", 3000)
            else:
                self.statusBar().showMessage(f"Failed to reload: {module_path}", 3000)
    
    def reload_home_page(self):
        """Reload home page widget"""
        try:
            # Get main layout
            central_widget = self.centralWidget()
            main_layout = central_widget.layout()
            
            # Remove old widget
            old_widget = self.home_widget
            main_layout.removeWidget(old_widget)
            
            # Xóa widget cũ để giải phóng bộ nhớ
            if old_widget:
                old_widget.deleteLater()
            
            # Force reload module
            if 'pages.home.home_page' in sys.modules:
                del sys.modules['pages.home.home_page']
            
            # Create new widget
            from pages.home.home_page import HomePage
            self.home_widget = HomePage()
            main_layout.addWidget(self.home_widget, 1)
            
            self.logger.info("HomePage reloaded successfully")
            self.statusBar().showMessage("Trang đã được tải lại!", 2000)
        except Exception as e:
            self.logger.error(f"Error reloading HomePage: {e}", exc_info=True)
            self.statusBar().showMessage(f"Reload failed: {str(e)}", 3000)
        
    def refresh_ui(self):
        """Refresh UI when theme changes"""
        # Re-apply styles or refresh widgets if needed
        pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.hotload_manager:
            self.hotload_manager.stop()
        event.accept()