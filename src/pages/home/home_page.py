from PySide6.QtWidgets import QWidget, QHBoxLayout, QSplitter
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
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Splitter để cho phép resize động
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: var(--border);
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: var(--primary);
            }
        """)
        
        # Canvas Panel (70%)
        self.canvas_panel = CanvasPanel()
        self.main_splitter.addWidget(self.canvas_panel)
        
        # Control Panel (30%)
        self.control_panel = ControlPanel()
        self.main_splitter.addWidget(self.control_panel)
        
        # Set initial sizes: 70% canvas, 30% control
        total_width = 1000  # giá trị tham chiếu
        self.main_splitter.setSizes([int(total_width * 0.7), int(total_width * 0.3)])
        
        # Set minimum width cho control panel
        self.control_panel.setMinimumWidth(300)
        
        main_layout.addWidget(self.main_splitter)
    
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
        self.canvas_panel.ocr_completed.connect(self.control_panel.on_ocr_result)
        
        # Connect panel visibility signals
        self.canvas_panel.panel_visibility_changed.connect(self.on_canvas_panel_visibility_changed)
        
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
    
    def on_canvas_panel_visibility_changed(self, left_visible: bool, right_visible: bool):
        """Handle khi visibility của canvas panels thay đổi"""
        # Tính toán lại sizes cho splitter
        current_sizes = self.main_splitter.sizes()
        total_width = sum(current_sizes)
        
        if not left_visible and not right_visible:
            # Cả 2 panels đều ẩn -> Canvas Panel thu nhỏ tối đa
            new_canvas_width = int(total_width * 0.05)  # 5% cho toggle bar
            new_control_width = int(total_width * 0.95)  # 95% cho control
        elif not left_visible or not right_visible:
            # 1 panel ẩn -> Canvas Panel trung bình
            new_canvas_width = int(total_width * 0.45)  # 45%
            new_control_width = int(total_width * 0.55)  # 55%
        else:
            # Cả 2 panels đều hiện -> Tỷ lệ ban đầu
            new_canvas_width = int(total_width * 0.7)  # 70%
            new_control_width = int(total_width * 0.3)  # 30%
        
        self.main_splitter.setSizes([new_canvas_width, new_control_width])