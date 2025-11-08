from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
import logging

from .widgets.canvas_panel.canvas_panel import CanvasPanel
from .widgets.control_panel.control_panel import ControlPanel


class HomePage(QWidget):
    """Trang chủ với Canvas Panel và Control Panel"""
    
    # Signals
    folder_selected = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        self.setup_shortcuts()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI cho trang chủ"""
        # Main layout - horizontal split
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Canvas Panel (70%)
        self.canvas_panel = CanvasPanel()
        main_layout.addWidget(self.canvas_panel, 7)
        
        # Control Panel (30%)
        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, 3)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Ctrl+O to open folder
        self.open_folder_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        self.open_folder_shortcut.activated.connect(self.on_open_folder)
            
    def setup_connections(self):
        """Setup signal connections"""
        # Connect canvas panel signals
        self.canvas_panel.folder_selected.connect(self.on_folder_selected)
        self.canvas_panel.image_changed.connect(self.control_panel.on_image_changed)
        self.canvas_panel.segmentation_completed.connect(self.on_processing_step_completed)
        self.canvas_panel.ocr_completed.connect(self.control_panel.on_ocr_result)  # THÊM DÒNG NÀY
        
        # Connect control panel signals
        self.control_panel.run_model_requested.connect(self.on_run_model)
        
        # Connect internal signals
        self.folder_selected.connect(self.control_panel.on_folder_loaded)
    
    def on_processing_step_completed(self, index: int, result_image):
        """Handle khi hoàn thành một bước xử lý"""
        # Có thể thêm logic cập nhật UI ở đây nếu cần
        pass
    
    def on_open_folder(self):
        """Handle Ctrl+O shortcut"""
        self.canvas_panel.open_folder_dialog()
    
    def on_folder_selected(self, folder_path: str):
        """Handle folder selection"""
        self.folder_selected.emit(folder_path)
    
    def on_run_model(self):
        """Handle run model request"""
        self.canvas_panel.start_segmentation()