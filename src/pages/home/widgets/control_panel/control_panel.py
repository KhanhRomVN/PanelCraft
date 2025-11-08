from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QTextEdit
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
        self.ocr_results: dict = {}  # {index: list of text}
        self.current_image_index: int = 0
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI với phần hiển thị OCR results"""
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
        
        # Model info
        model_info = QLabel(
            "Models:\n"
            "- YOLOv8 Segmentation\n" 
            "- Comic Text Detector\n"
            "- Manga OCR"
        )
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
            text="Run Full Pipeline",
            variant="primary",
            size="md"
        )
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self.on_run_clicked)
        model_layout.addWidget(self.run_button)
        
        # Progress label - THÊM CODE MỚI
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
        
        # OCR Results group
        ocr_group = QGroupBox("OCR Results")
        ocr_group.setStyleSheet("""
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
        
        ocr_layout = QVBoxLayout(ocr_group)
        ocr_layout.setSpacing(8)
        
        # Current image info
        self.current_image_label = QLabel("No image selected")
        self.current_image_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: normal;
            padding: 4px;
        """)
        ocr_layout.addWidget(self.current_image_label)
        
        # OCR results display
        self.ocr_text_edit = QTextEdit()
        self.ocr_text_edit.setStyleSheet("""
            QTextEdit {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        self.ocr_text_edit.setReadOnly(True)
        self.ocr_text_edit.setMaximumHeight(150)
        self.ocr_text_edit.setPlaceholderText("OCR results will appear here after processing...")
        ocr_layout.addWidget(self.ocr_text_edit)
        
        layout.addWidget(ocr_group)
        
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
            "• Click 'Run Full Pipeline' to process\n"
            "• Right panel shows text detection results\n"
            "• OCR results appear below"
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
        self.run_model_requested.emit()
        
        # Update status
        self.status_label.setText("Processing images...")
        self.run_button.setEnabled(False)
        self.progress_label.setText("Initializing pipeline...")
        self.progress_label.show()
        self.ocr_text_edit.setPlainText("Processing...")
    
    def on_folder_loaded(self, folder_path: str):
        """Handle folder loaded"""
        self.folder_loaded = True
        self.run_button.setEnabled(True)
        
        # Update status
        folder_name = folder_path.split('/')[-1] or folder_path
        self.status_label.setText(f"Folder loaded: {folder_name}")
        
    
    def on_ocr_result(self, index: int, texts: list):
        """Handle OCR results"""
        
        # Store result
        self.ocr_results[index] = texts
        
        # Update OCR display if this is the current image
        if index == self.current_image_index:
            self.update_ocr_display(texts)
            
    def on_image_changed(self, index: int):
        """Handle image navigation"""
        
        self.current_image_index = index
        self.update_current_image_label(index)
                
        # Update OCR display for current image
        if index in self.ocr_results:
            self.update_ocr_display(self.ocr_results[index])
        else:
            self.logger.warning(f"[CONTROL] No OCR results for index {index} yet")
            self.ocr_text_edit.setPlainText("No OCR results for this image yet.")
            
    def update_current_image_label(self, index: int):
        """Update current image label"""
        self.current_image_label.setText(f"Current Image: {index + 1}")
    
    def update_ocr_display(self, texts: list):
        """Update OCR text display"""
        
        if texts and any(text.strip() for text in texts):
            ocr_text = "\n".join([f"• {text}" for text in texts if text.strip()])
            self.ocr_text_edit.setPlainText(ocr_text)
        else:
            self.ocr_text_edit.setPlainText("No text detected")
            self.logger.warning(f"[CONTROL] No valid text to display")
    
    def on_processing_complete(self):
        """Handle processing complete"""
        self.status_label.setText("Processing complete!")
        self.run_button.setEnabled(True)
        self.progress_label.hide()
