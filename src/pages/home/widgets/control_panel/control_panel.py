from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
                              QTableWidget, QTableWidgetItem, QComboBox, QHeaderView, QFrame)
from PySide6.QtCore import Qt, Signal
import logging

from widget.common.custom_button import CustomButton
from widget.common.custom_table import CustomTable
from widget.common.custom_input import CustomInput


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
        
        # OCR Results group
        ocr_group = QGroupBox("OCR Results & Translation")
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
        
        # Current image info & Character management
        header_layout = QHBoxLayout()
        
        self.current_image_label = QLabel("No image selected")
        self.current_image_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: normal;
            padding: 4px;
        """)
        header_layout.addWidget(self.current_image_label)
        
        header_layout.addStretch()
        
        # OCR Region Selection button
        self.ocr_select_btn = CustomButton(
            text="🔍 Select OCR Region",
            variant="secondary",
            size="sm"
        )
        self.ocr_select_btn.setCheckable(True)
        self.ocr_select_btn.clicked.connect(self.on_toggle_ocr_mode)
        header_layout.addWidget(self.ocr_select_btn)
        
        # Manage Characters button
        self.manage_characters_btn = CustomButton(
            text="Manage Characters",
            variant="secondary",
            size="sm"
        )
        self.manage_characters_btn.clicked.connect(self.on_manage_characters)
        header_layout.addWidget(self.manage_characters_btn)
        
        ocr_layout.addLayout(header_layout)
        
        # OCR results table - Dùng CustomTable
        self.ocr_table = CustomTable(
            headers=["STT", "Character", "Original Text", "Translation"],
            page_size=15,
            show_pagination=False
        )
        
        # Style cho table (CustomTable tự xử lý phần lớn)
        ocr_layout.addWidget(self.ocr_table)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: var(--border);")
        ocr_layout.addWidget(separator)
        
        # Edit section - Hiển thị khi click vào row
        self.edit_container = QWidget()
        self.edit_container.hide()
        edit_layout = QVBoxLayout(self.edit_container)
        edit_layout.setContentsMargins(0, 8, 0, 0)
        edit_layout.setSpacing(12)
        
        # Original text input
        self.original_input = CustomInput(
            label="Original Text",
            placeholder="Original text from OCR...",
            variant="filled",
            multiline=True,
            rows=3
        )
        self.original_input.input_field.setReadOnly(False)
        edit_layout.addWidget(self.original_input)
        
        # Translation input
        self.translation_input = CustomInput(
            label="Translation",
            placeholder="Enter translation here...",
            variant="filled",
            multiline=True,
            rows=3
        )
        edit_layout.addWidget(self.translation_input)
        
        # Save button
        save_btn_layout = QHBoxLayout()
        save_btn_layout.addStretch()
        self.save_edit_btn = CustomButton(
            text="Save Changes",
            variant="primary",
            size="sm"
        )
        self.save_edit_btn.clicked.connect(self.on_save_edit)
        save_btn_layout.addWidget(self.save_edit_btn)
        edit_layout.addLayout(save_btn_layout)
        
        ocr_layout.addWidget(self.edit_container)
        
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
        self.ocr_table.setRowCount(0)  # Clear table
    
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
            self.ocr_table.setRowCount(0)
            
    def update_current_image_label(self, index: int):
        """Update current image label"""
        self.current_image_label.setText(f"Current Image: {index + 1}")
    
    def update_ocr_display(self, texts: list):
        """Update OCR table display với CustomTable"""
        from core.project_manager import ProjectManager
        
        if not texts or not any(text.strip() for text in texts):
            self.ocr_table.setData([])
            self.logger.warning(f"[CONTROL] No valid text to display")
            return
        
        project_manager = ProjectManager()
        characters = project_manager.get_characters()
        
        # Prepare data for CustomTable
        valid_texts = [text for text in texts if text.strip()]
        table_data = []
        
        self.logger.info(f"[OCR TABLE] Displaying {len(valid_texts)} texts in manga reading order")
        
        for i, text in enumerate(valid_texts):
            # Truncate text nếu quá dài
            original_short = text if len(text) <= 50 else text[:47] + "..."
            
            table_data.append({
                "STT": str(i + 1),
                "Character": "-- Select --",
                "Original Text": original_short,
                "Translation": "",
                "_full_original": text,  # Lưu full text để edit
                "_full_translation": "",
                "_character_id": None
            })
        
        self.ocr_table.setData(table_data)
        
        # Lưu data gốc để edit
        self.ocr_data = table_data
        
    def on_row_clicked(self, row_index: int, row_data: dict):
        """Handle khi click vào row trong table"""
        
        # Hiển thị edit container
        self.edit_container.show()
        
        # Load data vào inputs
        self.original_input.setText(row_data.get("_full_original", ""))
        self.translation_input.setText(row_data.get("_full_translation", ""))
        
        # Lưu row index đang edit
        self.current_edit_row = row_index
        
        self.logger.info(f"[CONTROL] Editing row {row_index}")
    
    def on_save_edit(self):
        """Lưu thay đổi từ edit inputs vào table data"""
        if not hasattr(self, 'current_edit_row'):
            return
        
        row = self.current_edit_row
        
        # Lấy text từ inputs
        full_original = self.original_input.text()
        full_translation = self.translation_input.text()
        
        # Cập nhật data
        if row < len(self.ocr_data):
            self.ocr_data[row]["_full_original"] = full_original
            self.ocr_data[row]["_full_translation"] = full_translation
            
            # Cập nhật display (truncated)
            original_short = full_original if len(full_original) <= 50 else full_original[:47] + "..."
            translation_short = full_translation if len(full_translation) <= 50 else full_translation[:47] + "..."
            
            self.ocr_data[row]["Original Text"] = original_short
            self.ocr_data[row]["Translation"] = translation_short
            
            # Refresh table
            self.ocr_table.setData(self.ocr_data)
            
            self.logger.info(f"[CONTROL] Saved changes for row {row}")
    
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
    
    def on_character_changed(self, row: int, combo_index: int):
        """Handle khi user thay đổi character cho một dòng"""
        combo = self.ocr_table.cellWidget(row, 1)
        if combo:
            char_id = combo.currentData()
            char_name = combo.currentText()
            
            if char_id:
                self.logger.info(f"Row {row}: Character changed to {char_name} (ID: {char_id})")
            else:
                self.logger.info(f"Row {row}: No character selected")
    
    def get_ocr_table_data(self) -> list:
        """Lấy dữ liệu từ table để lưu vào project"""
        if not hasattr(self, 'ocr_data'):
            return []
        
        data = []
        for i, row in enumerate(self.ocr_data):
            data.append({
                'index': i,
                'character_id': row.get('_character_id'),
                'original_text': row.get('_full_original', ''),
                'translated_text': row.get('_full_translation', '')
            })
        
        return data