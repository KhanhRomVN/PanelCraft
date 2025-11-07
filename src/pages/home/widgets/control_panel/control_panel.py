from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox
from PySide6.QtCore import Qt, Signal
import logging

from widget.common.custom_button import CustomButton


class ControlPanel(QWidget):
    """Panel điều khiển với các controls"""
    
    # Signals
    run_model_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.folder_loaded = False
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Controls")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            padding: 8px 0;
        """)
        layout.addWidget(title)
        
        # Model group
        model_group = QGroupBox("Segmentation Model")
        model_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid var(--border);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: var(--card-background);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: var(--text-primary);
            }
        """)
        
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(12)
        
        # Model info
        model_info = QLabel("Model: YOLOv8 Segmentation\nFile: yolov8_converted.onnx")
        model_info.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: normal;
            padding: 8px;
            background-color: var(--sidebar-background);
            border-radius: 4px;
        """)
        model_info.setWordWrap(True)
        model_layout.addWidget(model_info)
        
        # Run button
        self.run_button = CustomButton(
            text="Run Segmentation",
            variant="primary",
            size="md"
        )
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.on_run_clicked)
        model_layout.addWidget(self.run_button)
        
        layout.addWidget(model_group)
        
        # Status group
        status_group = QGroupBox("Status")
        status_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid var(--border);
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: var(--card-background);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: var(--text-primary);
            }
        """)
        
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        
        self.status_label = QLabel("No folder loaded")
        self.status_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: normal;
            padding: 8px;
        """)
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
        # Instructions
        instructions_label = QLabel(
            "Instructions:\n"
            "• Press Ctrl+O to open folder\n"
            "• Use ↑↓ keys to navigate images\n"
            "• Click 'Run Segmentation' to process"
        )
        instructions_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            padding: 12px;
            background-color: var(--sidebar-background);
            border-radius: 6px;
            line-height: 1.6;
        """)
        instructions_label.setWordWrap(True)
        layout.addWidget(instructions_label)
        
        # Spacer
        layout.addStretch()
    
    def on_run_clicked(self):
        """Handle run button click"""
        self.logger.info("Run segmentation clicked")
        self.run_model_requested.emit()
        
        # Update status
        self.status_label.setText("Processing images...")
        self.run_button.setEnabled(False)
    
    def on_folder_loaded(self, folder_path: str):
        """Handle folder loaded"""
        self.folder_loaded = True
        self.run_button.setEnabled(True)
        
        # Update status
        folder_name = folder_path.split('/')[-1] or folder_path
        self.status_label.setText(f"Folder loaded: {folder_name}")
        
        self.logger.info(f"Folder loaded in control panel: {folder_path}")
    
    def on_processing_complete(self):
        """Handle processing complete"""
        self.status_label.setText("Processing complete!")
        self.run_button.setEnabled(True)