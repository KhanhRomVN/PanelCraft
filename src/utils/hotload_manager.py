import logging
import os
import importlib
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PySide6.QtCore import QObject, Signal


class HotloadHandler(FileSystemEventHandler):
    """Handler for file system events"""
    
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        
    def on_modified(self, event):
        """Called when a file is modified"""
        if event.is_directory:
            return
            
        if event.src_path.endswith('.py'):
            self.logger.info(f"Detected change in: {event.src_path}")
            self.callback(event.src_path)


class HotloadManager(QObject):
    """Manager for hotloading widgets and modules"""
    
    reload_requested = Signal(str)
    
    def __init__(self, watch_paths=None, enabled=True):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self.observer = None
        self.watch_paths = watch_paths or []
        
        if self.enabled:
            self.setup_watcher()
    
    def setup_watcher(self):
        """Setup file system watcher"""
        try:
            self.observer = Observer()
            handler = HotloadHandler(self.on_file_changed)
            
            for path in self.watch_paths:
                if os.path.exists(path):
                    self.observer.schedule(handler, path, recursive=True)
                    self.logger.info(f"Watching path: {path}")
                else:
                    self.logger.warning(f"Path does not exist: {path}")
            
            self.observer.start()
            self.logger.info("Hotload watcher started")
        except Exception as e:
            self.logger.error(f"Failed to setup watcher: {e}")
    
    def on_file_changed(self, file_path):
        """Called when a file is changed"""
        if not self.enabled:
            return
        
        try:
            # Convert file path to module path
            module_path = self.file_path_to_module(file_path)
            if module_path:
                self.reload_requested.emit(module_path)
        except Exception as e:
            self.logger.error(f"Error processing file change: {e}")
    
    def file_path_to_module(self, file_path):
        """Convert file path to Python module path"""
        try:
            # Get relative path from src directory
            file_path = Path(file_path).resolve()
            
            # Tìm thư mục src trong đường dẫn
            parts = file_path.parts
            try:
                src_index = parts.index('src')
                # Lấy các phần sau 'src'
                module_parts = list(parts[src_index + 1:])
                
                # Bỏ phần mở rộng .py
                if module_parts[-1].endswith('.py'):
                    module_parts[-1] = module_parts[-1][:-3]
                
                # Bỏ __init__ nếu có
                if module_parts[-1] == '__init__':
                    module_parts.pop()
                
                module_path = '.'.join(module_parts)
                self.logger.debug(f"Converted {file_path} -> {module_path}")
                return module_path
            except ValueError:
                self.logger.error(f"'src' not found in path: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting path to module: {e}")
            return None
    
    def reload_module(self, module_path):
        """Reload a Python module"""
        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]
                importlib.reload(module)
                self.logger.info(f"Reloaded module: {module_path}")
                return True
            else:
                self.logger.warning(f"Module not loaded: {module_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error reloading module {module_path}: {e}")
            return False
    
    def stop(self):
        """Stop the file watcher"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Hotload watcher stopped")
    
    def set_enabled(self, enabled):
        """Enable or disable hotloading"""
        self.enabled = enabled
        if enabled and not self.observer:
            self.setup_watcher()
        elif not enabled and self.observer:
            self.stop()