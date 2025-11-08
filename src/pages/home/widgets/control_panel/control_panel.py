from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, 
                              QTableWidget, QTableWidgetItem, QComboBox, QHeaderView)
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
        
        # Pipeline description
        pipeline_info = QLabel(
            "5-step processing:\n"
            "1. Segmentation (YOLOv8)\n"
            "2. Visualization with rectangles\n"
            "3. OCR text recognition\n"
            "4. Text detection & removal\n"
            "5. Final composition"
        )
        pipeline_info.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: normal;
            padding: 8px;
            background-color: var(--sidebar-background);
            border-radius: 4px;
        """)
        pipeline_info.setWordWrap(True)
        model_layout.addWidget(pipeline_info)
        
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
        
        # Manage Characters button
        self.manage_characters_btn = CustomButton(
            text="Manage Characters",
            variant="secondary",
            size="sm"
        )
        self.manage_characters_btn.clicked.connect(self.on_manage_characters)
        header_layout.addWidget(self.manage_characters_btn)
        
        ocr_layout.addLayout(header_layout)
        
        # OCR results table
        self.ocr_table = QTableWidget()
        self.ocr_table.setColumnCount(4)
        self.ocr_table.setHorizontalHeaderLabels(["STT", "Character", "Bản gốc", "Bản dịch"])
        
        # Set column widths
        self.ocr_table.setColumnWidth(0, 50)   # STT
        self.ocr_table.setColumnWidth(1, 120)  # Character
        self.ocr_table.setColumnWidth(2, 200)  # Bản gốc
        self.ocr_table.setColumnWidth(3, 200)  # Bản dịch
        
        # Enable stretch for last column
        header = self.ocr_table.horizontalHeader()
        header.setStretchLastSection(True)
        
        # Enable text wrapping và auto-resize rows
        self.ocr_table.setWordWrap(True)
        self.ocr_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        self.ocr_table.setStyleSheet("""
            QTableWidget {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                gridline-color: var(--border);
            }
            QTableWidget::item {
                padding: 8px;
                line-height: 1.4;
            }
            QHeaderView::section {
                background-color: var(--sidebar-background);
                color: var(--text-primary);
                padding: 6px;
                border: 1px solid var(--border);
                font-weight: bold;
            }
        """)
        
        # Set minimum height để hiển thị tối đa 10 rows
        # Row height ước tính: ~50px (với text wrapping)
        # Header height: ~35px
        # Scrollbar + padding: ~10px
        self.ocr_table.setMinimumHeight(200)  # Tối thiểu 4-5 rows
        self.ocr_table.setMaximumHeight(550)  # Tối đa ~10 rows
        
        ocr_layout.addWidget(self.ocr_table)
        
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
            "• Assign characters and add translations\n"
            "• Manage characters via button above"
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
        """Update OCR table display với danh sách texts"""
        from core.project_manager import ProjectManager
        
        if not texts or not any(text.strip() for text in texts):
            self.ocr_table.setRowCount(0)
            self.logger.warning(f"[CONTROL] No valid text to display")
            return
        
        project_manager = ProjectManager()
        characters = project_manager.get_characters()
        
        # Clear table
        self.ocr_table.setRowCount(0)
        
        # Populate table
        valid_texts = [text for text in texts if text.strip()]
        
        for i, text in enumerate(valid_texts):
            row_position = self.ocr_table.rowCount()
            self.ocr_table.insertRow(row_position)
            
            # Column 0: STT
            stt_item = QTableWidgetItem(str(i + 1))
            stt_item.setFlags(stt_item.flags() & ~Qt.ItemIsEditable)  # Read-only
            self.ocr_table.setItem(row_position, 0, stt_item)
            
            # Column 1: Character ComboBox
            char_combo = QComboBox()
            char_combo.addItem("-- Chọn nhân vật --", None)
            
            for char in characters:
                char_combo.addItem(char['name'], char['id'])
            
            char_combo.currentIndexChanged.connect(
                lambda idx, row=row_position: self.on_character_changed(row, idx)
            )
            
            self.ocr_table.setCellWidget(row_position, 1, char_combo)
            
            # Column 2: Bản gốc (Read-only)
            original_item = QTableWidgetItem(text)
            original_item.setFlags(original_item.flags() & ~Qt.ItemIsEditable)
            self.ocr_table.setItem(row_position, 2, original_item)
            
            # Column 3: Bản dịch (Editable)
            translated_item = QTableWidgetItem("")
            self.ocr_table.setItem(row_position, 3, translated_item)
        
        # Tự động điều chỉnh chiều cao table sau khi populate
        self.adjust_table_height()
    
    def adjust_table_height(self):
        """Tự động điều chỉnh chiều cao table dựa trên số rows"""
        row_count = self.ocr_table.rowCount()
        
        if row_count == 0:
            self.ocr_table.setMinimumHeight(200)
            return
        
        # Tính chiều cao header
        header_height = self.ocr_table.horizontalHeader().height()
        
        # Tính tổng chiều cao của các rows (tối đa 10 rows)
        total_row_height = 0
        visible_rows = min(row_count, 10)
        
        for row in range(visible_rows):
            total_row_height += self.ocr_table.rowHeight(row)
        
        # Tổng chiều cao = header + rows + margin
        total_height = header_height + total_row_height + 20
        
        # Giới hạn chiều cao
        min_height = 200
        max_height = 550
        
        calculated_height = max(min_height, min(total_height, max_height))
        
        self.ocr_table.setMinimumHeight(calculated_height)
    
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
    
    def refresh_character_comboboxes(self):
        """Cập nhật lại tất cả ComboBox character trong table"""
        from core.project_manager import ProjectManager
        
        project_manager = ProjectManager()
        characters = project_manager.get_characters()
        
        for row in range(self.ocr_table.rowCount()):
            combo = self.ocr_table.cellWidget(row, 1)
            if combo:
                current_char = combo.currentText()
                combo.clear()
                combo.addItem("-- Chọn nhân vật --", None)
                
                for char in characters:
                    combo.addItem(char['name'], char['id'])
                
                # Khôi phục lựa chọn cũ nếu còn tồn tại
                index = combo.findText(current_char)
                if index >= 0:
                    combo.setCurrentIndex(index)
    
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
        data = []
        
        for row in range(self.ocr_table.rowCount()):
            combo = self.ocr_table.cellWidget(row, 1)
            original_item = self.ocr_table.item(row, 2)
            translated_item = self.ocr_table.item(row, 3)
            
            char_id = combo.currentData() if combo else None
            original_text = original_item.text() if original_item else ""
            translated_text = translated_item.text() if translated_item else ""
            
            data.append({
                'index': row,
                'character_id': char_id,
                'original_text': original_text,
                'translated_text': translated_text
            })
        
        return data