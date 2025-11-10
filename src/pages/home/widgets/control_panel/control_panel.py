from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
                              QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QFrame)
from PySide6.QtCore import Qt, Signal
import logging

from widget.common.custom_button import CustomButton
from widget.common.custom_table import CustomTable
from widget.common.custom_input import CustomInput
from ...models.ocr_result import OCRResult
from ...constants.constants import OCR_DISPLAY_MAX_LENGTH
from ...types.home_types import OCRTableData
from .ocr_results_table import OCRResultsTable


class ControlPanel(QWidget):
    """Panel điều khiển với các controls"""
    
    # Signals
    run_model_requested = Signal()
    ocr_mode_toggled = Signal(bool)  # True = enable, False = disable
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.folder_loaded = False
        self.ocr_results: dict = {}  # {index: list of text}
        self.current_image_index: int = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI với phần hiển thị OCR results"""
        # Tạo main layout chứa scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Tạo scroll area
        from PySide6.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Tạo container widget cho nội dung
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
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
        model_group = QGroupBox("Processing Pipeline")
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
    
        
        # Run button
        self.run_button = CustomButton(
            text="Run Full Pipeline",
            variant="primary",
            size="md"
        )
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.on_run_clicked)
        model_layout.addWidget(self.run_button)
        
        # Progress label
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            padding: 4px;
            font-style: italic;
        """)
        self.progress_label.setWordWrap(True)
        self.progress_label.hide()
        model_layout.addWidget(self.progress_label)
        
        layout.addWidget(model_group)
        
        # OCR Results Table
        self.ocr_results_table = OCRResultsTable()
        self.ocr_results_table.ocr_mode_toggled.connect(self.on_ocr_mode_toggled)
        self.ocr_results_table.manage_characters_requested.connect(self.on_manage_characters)
        layout.addWidget(self.ocr_results_table)
        
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
                
        # Spacer
        layout.addStretch()
        
        # Set content widget vào scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area vào main layout
        main_layout.addWidget(scroll_area)
        
        # Set content widget vào scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area vào main layout
        main_layout.addWidget(scroll_area)
    
    def on_run_clicked(self):
        """Handle run button click"""
        self.run_model_requested.emit()
        
        # Update status
        self.status_label.setText("Processing images...")
        self.run_button.setEnabled(False)
        self.progress_label.setText("Initializing pipeline...")
        self.progress_label.show()
        self.ocr_results_table.clear_table()  # Clear table
    
    def on_folder_loaded(self, folder_path: str):
        """Handle folder loaded"""
        self.folder_loaded = True
        self.run_button.setEnabled(True)
        
        # Update status
        folder_name = folder_path.split('/')[-1] or folder_path
        self.status_label.setText(f"Folder loaded: {folder_name}")
        
    def on_ocr_mode_toggled(self, enabled: bool):
        """Toggle OCR mode - forward to parent"""
        self.ocr_mode_toggled.emit(enabled)
    
    def on_ocr_result(self, index: int, texts: list):
        # Forward to OCR results table
        self.ocr_results_table.on_ocr_result(index, texts)
            
    def on_image_changed(self, index: int):
        """Handle image navigation"""
        # Forward to OCR results table
        self.ocr_results_table.on_image_changed(index)
    
    def on_processing_complete(self):
        """Handle processing complete"""
        self.status_label.setText("Processing complete!")
        self.run_button.setEnabled(True)
        self.progress_label.hide()
        
    def on_manage_characters(self):
        """Mở dialog quản lý characters"""
        from .character_manager_dialog import CharacterManagerDialog
        
        dialog = CharacterManagerDialog(self)
        if dialog.exec():
            # Cập nhật lại ComboBox trong table sau khi thay đổi characters
            self.refresh_character_comboboxes()
            
    def on_ocr_region_result(self, text: str):
        """
        Nhận kết quả OCR từ region selection và update vào row đang focus
        
        Args:
            text: Kết quả OCR
        """        
        # Forward đến OCR results table
        self.ocr_results_table.on_ocr_region_result(text)
    
    def on_toggle_ocr_mode(self, checked: bool):
        """Toggle OCR selection mode"""
        # Emit signal để HomePage forward đến CanvasPanel
        if hasattr(self, 'ocr_mode_toggled'):
            self.ocr_mode_toggled.emit(checked)
        
        # Update button style
        if checked:
            self.ocr_select_btn.setText("✓ OCR Mode Active")
            self.ocr_select_btn.setStyleSheet("""
                QPushButton {
                    background-color: var(--primary);
                    color: white;
                    border: 1px solid var(--primary);
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                    min-width: 120px;
                }
            """)
        else:
            self.ocr_select_btn.setText("🔍 Select OCR Region")
            self.ocr_select_btn.setStyleSheet("""
                QPushButton {
                    background-color: var(--button-second-bg);
                    color: var(--text-primary);
                    border: 1px solid var(--border);
                    border-radius: 6px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: 500;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background-color: var(--button-second-bg-hover);
                }
            """)