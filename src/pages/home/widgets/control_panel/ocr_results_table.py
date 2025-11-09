from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QGroupBox, QFrame, QComboBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
import logging

from widget.common.custom_button import CustomButton
from widget.common.custom_table import CustomTable
from widget.common.custom_input import CustomInput
from core.project_manager import ProjectManager
from ...constants.constants import OCR_DISPLAY_MAX_LENGTH


class OCRResultsTable(QWidget):
    """Widget quản lý OCR results table và editing"""
    
    # Signals
    ocr_mode_toggled = Signal(bool)
    manage_characters_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.ocr_results = {}  # {index: list of text}
        self.current_image_index = 0
        self.ocr_data = []  # Current table data
        self.current_edit_row = None
        self.current_detail_row = None  # THÊM: Track row hiện tại trong detail view
        self.setup_ui()
        self.setup_table_connections()  # THÊM: Setup connections sau khi UI ready
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
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
            text="👥 Manage Characters",
            variant="secondary",
            size="sm"
        )
        self.manage_characters_btn.setEnabled(False)  # THÊM: Mặc định disable
        self.manage_characters_btn.clicked.connect(self.on_manage_characters)
        header_layout.addWidget(self.manage_characters_btn)
        
        # Manage Fonts button
        self.manage_fonts_btn = CustomButton(
            text="🔤 Manage Fonts",
            variant="secondary",
            size="sm"
        )
        self.manage_fonts_btn.clicked.connect(self.on_manage_fonts)
        header_layout.addWidget(self.manage_fonts_btn)
        
        ocr_layout.addLayout(header_layout)
        
        # OCR results table
        self.ocr_table = CustomTable(
            headers=["STT", "Character", "Font", "Original Text", "Translation"],
            page_size=15,
            show_pagination=False
        )
        
        # ẨN vertical header (cột index bên trái)
        self.ocr_table.table_widget.verticalHeader().setVisible(False)
        
        ocr_layout.addWidget(self.ocr_table)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: var(--border);")
        ocr_layout.addWidget(separator)
        
        # Edit section
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
        
        # ========== Detail View Panel ==========
        # Separator line
        detail_separator = QFrame()
        detail_separator.setFrameShape(QFrame.HLine)
        detail_separator.setFrameShadow(QFrame.Sunken)
        detail_separator.setStyleSheet("background-color: var(--border);")
        ocr_layout.addWidget(detail_separator)
        
        # Detail container
        self.detail_container = QWidget()
        self.detail_container.hide()
        detail_layout = QVBoxLayout(self.detail_container)
        detail_layout.setContentsMargins(0, 8, 0, 0)
        detail_layout.setSpacing(12)
        
        # Title
        detail_title = QLabel("Chi tiết OCR")
        detail_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: var(--text-primary);
            padding: 8px 0;
        """)
        detail_layout.addWidget(detail_title)
        
        # STT field (read-only)
        self.detail_stt_input = CustomInput(
            label="STT",
            variant="filled",
            size="sm"
        )
        self.detail_stt_input.input_field.setReadOnly(True)
        detail_layout.addWidget(self.detail_stt_input)
        
        # Character field (CustomCombobox)
        from widget.common.custom_combobox import CustomCombobox
        
        self.detail_character_combo = CustomCombobox(
            label="Nhân vật",
            placeholder="-- Chọn nhân vật --",
            searchable=True,
            size="sm"
        )
        detail_layout.addWidget(self.detail_character_combo)
        
        # Original text field (read-only, multiline)
        self.detail_original_input = CustomInput(
            label="Văn bản gốc",
            variant="filled",
            multiline=True,
            rows=4
        )
        self.detail_original_input.input_field.setReadOnly(True)
        detail_layout.addWidget(self.detail_original_input)
        
        # Translation field (editable, multiline)
        self.detail_translation_input = CustomInput(
            label="Bản dịch",
            placeholder="Nhập bản dịch tại đây...",
            variant="filled",
            multiline=True,
            rows=4
        )
        detail_layout.addWidget(self.detail_translation_input)
        
        # Action buttons
        detail_btn_layout = QHBoxLayout()
        detail_btn_layout.addStretch()
        
        self.detail_close_btn = CustomButton(
            text="Đóng",
            variant="secondary",
            size="sm"
        )
        self.detail_close_btn.clicked.connect(self.on_close_detail)
        detail_btn_layout.addWidget(self.detail_close_btn)
        
        self.detail_save_btn = CustomButton(
            text="Lưu thay đổi",
            variant="primary",
            size="sm"
        )
        self.detail_save_btn.clicked.connect(self.on_save_detail)
        detail_btn_layout.addWidget(self.detail_save_btn)
        
        detail_layout.addLayout(detail_btn_layout)
        
        ocr_layout.addWidget(self.detail_container)
        
        layout.addWidget(ocr_group)
        detail_separator.setFrameShape(QFrame.HLine)
        detail_separator.setFrameShadow(QFrame.Sunken)
        detail_separator.setStyleSheet("background-color: var(--border);")
        ocr_layout.addWidget(detail_separator)
        
        # Detail container
        self.detail_container = QWidget()
        self.detail_container.hide()
        detail_layout = QVBoxLayout(self.detail_container)
        detail_layout.setContentsMargins(0, 8, 0, 0)
        detail_layout.setSpacing(12)
        
        # Title
        detail_title = QLabel("Chi tiết OCR")
        detail_title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: var(--text-primary);
            padding: 8px 0;
        """)
        detail_layout.addWidget(detail_title)
        
        # STT field (read-only)
        self.detail_stt_input = CustomInput(
            label="STT",
            variant="filled",
            size="sm"
        )
        self.detail_stt_input.input_field.setReadOnly(True)
        detail_layout.addWidget(self.detail_stt_input)
        
        # Character field (CustomCombobox)
        from widget.common.custom_combobox import CustomCombobox
        
        self.detail_character_combo = CustomCombobox(
            label="Nhân vật",
            placeholder="-- Chọn nhân vật --",
            searchable=True,
            size="sm"
        )
        detail_layout.addWidget(self.detail_character_combo)
        
        # Font field (CustomCombobox)
        self.detail_font_combo = CustomCombobox(
            label="Font",
            placeholder="-- Chọn font --",
            searchable=True,
            size="sm"
        )
        detail_layout.addWidget(self.detail_font_combo)
        
        # Original text field (read-only, multiline)
        self.detail_original_input = CustomInput(
            label="Văn bản gốc",
            variant="filled",
            multiline=True,
            rows=4
        )
        self.detail_original_input.input_field.setReadOnly(True)
        detail_layout.addWidget(self.detail_original_input)
        
        # Translation field (editable, multiline)
        self.detail_translation_input = CustomInput(
            label="Bản dịch",
            placeholder="Nhập bản dịch tại đây...",
            variant="filled",
            multiline=True,
            rows=4
        )
        detail_layout.addWidget(self.detail_translation_input)
        
        # Action buttons
        detail_btn_layout = QHBoxLayout()
        detail_btn_layout.addStretch()
        
        self.detail_close_btn = CustomButton(
            text="Đóng",
            variant="secondary",
            size="sm"
        )
        self.detail_close_btn.clicked.connect(self.on_close_detail)
        detail_btn_layout.addWidget(self.detail_close_btn)
        
        self.detail_save_btn = CustomButton(
            text="Lưu thay đổi",
            variant="primary",
            size="sm"
        )
        self.detail_save_btn.clicked.connect(self.on_save_detail)
        detail_btn_layout.addWidget(self.detail_save_btn)
        
        detail_layout.addLayout(detail_btn_layout)
        
        ocr_layout.addWidget(self.detail_container)
        
        layout.addWidget(ocr_group)
    
    def setup_table_connections(self):
        """Setup connections cho table"""
        # Connect row click để hiển thị detail view VÀ kích hoạt OCR selection mode
        self.ocr_table.rowClicked.connect(self.on_row_clicked_detail)
        self.ocr_table.rowClicked.connect(self.on_row_clicked_activate_ocr_mode)
    
    def on_toggle_ocr_mode(self, checked: bool):
        """Toggle OCR selection mode"""
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
    
    def on_row_clicked_activate_ocr_mode(self, row_index: int, row_data: dict):
        """Kích hoạt OCR mode khi click vào row"""
        self.logger.info(f"[OCR_TABLE] Row {row_index} clicked - Activating OCR selection mode")
        
        # Store current focused row
        self.current_detail_row = row_index
        
        # Kích hoạt OCR mode nếu chưa active
        if not self.ocr_select_btn.isChecked():
            self.ocr_select_btn.setChecked(True)
            self.on_toggle_ocr_mode(True)
    
    def on_manage_characters(self):
        """Emit signal để mở dialog quản lý characters"""
        self.manage_characters_requested.emit()
        
    def on_manage_characters(self):
        """Emit signal để mở dialog quản lý characters"""
        from .character_manager_dialog import CharacterManagerDialog
        
        dialog = CharacterManagerDialog(self)
        if dialog.exec():
            # Cập nhật lại ComboBox trong table sau khi thay đổi characters
            self.refresh_character_comboboxes()
    
    def on_manage_fonts(self):
        """Mở dialog quản lý fonts"""
        from .font_manager_dialog import FontManagerDialog
        
        dialog = FontManagerDialog(self)
        if dialog.exec():
            # Cập nhật lại font comboboxes trong table
            self.refresh_font_comboboxes()
            self.logger.info("[OCR_TABLE] Font settings updated")
        
    def on_row_clicked_detail(self, row_index: int, row_data: dict):
        """Handle row click - hiển thị detail view"""
        self.logger.info(f"[OCR_TABLE] Row clicked: {row_index}")
        
        # Ẩn edit container (nếu đang mở)
        self.edit_container.hide()
        
        # Hiển thị detail container
        self.detail_container.show()
        
        # Store current row
        self.current_detail_row = row_index
        
        # Load fonts và characters vào ComboBox
        self._load_fonts_to_detail_combo()
        self._load_characters_to_detail_combo()
        
        # Load data vào detail view
        self.detail_stt_input.setText(row_data.get("STT", ""))
        
        # Set character
        char_id = row_data.get("_character_id")
        if char_id:
            self.detail_character_combo.setCurrentValue(char_id)
        else:
            self.detail_character_combo.setCurrentValue(None)
        
        # Set font
        current_font = row_data.get("_font_family")
        if current_font:
            self.detail_font_combo.setCurrentValue(current_font)
        else:
            self.detail_font_combo.setCurrentValue("default")
        
        self.detail_original_input.setText(row_data.get("_full_original", ""))
        self.detail_translation_input.setText(row_data.get("_full_translation", ""))
        
        self.logger.info(f"[OCR_TABLE] Detail view opened for row {row_index}")
        
    def _load_fonts_to_detail_combo(self):
        """Load fonts vào detail view CustomCombobox"""
        from core.font_manager import FontManager
        
        font_manager = FontManager()
        visible_fonts = font_manager.get_visible_fonts()
        default_font = font_manager.get_default_font()
        
        # Tạo options cho CustomCombobox
        font_options = [{"value": "default", "label": "Default"}]
        for font in visible_fonts:
            font_options.append({"value": font, "label": font})
        
        # Set options
        self.detail_font_combo.setOptions(font_options)
        
        # Set default font nếu có
        if default_font:
            self.detail_font_combo.setCurrentValue(default_font)
        else:
            self.detail_font_combo.setCurrentValue("default")
            
    def _load_characters_to_detail_combo(self):
        """Load characters vào detail view CustomCombobox"""
        from core.project_manager import ProjectManager
        
        project_manager = ProjectManager()
        characters = project_manager.get_characters()
        
        # Tạo options cho CustomCombobox
        char_options = [{"value": None, "label": "-- Chọn nhân vật --"}]
        for char in characters:
            char_options.append({
                "value": char['id'],
                "label": char['name']
            })
        
        # Set options
        self.detail_character_combo.setOptions(char_options)
        
    def on_close_detail(self):
        """Đóng detail view"""
        self.detail_container.hide()
        self.current_detail_row = None
        
        # Tắt OCR mode khi đóng detail
        if self.ocr_select_btn.isChecked():
            self.ocr_select_btn.setChecked(False)
            self.on_toggle_ocr_mode(False)
        
        self.logger.info("[OCR_TABLE] Detail view closed")
    
    def on_save_detail(self):
        """Lưu thay đổi từ detail view"""
        if self.current_detail_row is None:
            return
        
        row = self.current_detail_row
        
        # Get text from detail inputs
        full_translation = self.detail_translation_input.text()
        
        # Get selected font and character
        selected_font_value = self.detail_font_combo.currentValue()
        selected_char_value = self.detail_character_combo.currentValue()
        
        # Update data
        if row < len(self.ocr_data):
            self.ocr_data[row]["_full_translation"] = full_translation
            
            # Update font
            if selected_font_value and selected_font_value != "default":
                self.ocr_data[row]["_font_family"] = selected_font_value
                self.ocr_data[row]["Font"] = selected_font_value  # Display in table
            else:
                self.ocr_data[row]["_font_family"] = None
                self.ocr_data[row]["Font"] = "Default"
            
            # Update character
            if selected_char_value:
                self.ocr_data[row]["_character_id"] = selected_char_value
                # Get character name from combo
                char_text = self.detail_character_combo.combobox.currentText()
                self.ocr_data[row]["Character"] = char_text
            else:
                self.ocr_data[row]["_character_id"] = None
                self.ocr_data[row]["Character"] = "-- Select --"
            
            # Update display (truncated)
            max_len = OCR_DISPLAY_MAX_LENGTH
            translation_short = full_translation if len(full_translation) <= max_len else full_translation[:max_len - 3] + "..."
            
            self.ocr_data[row]["Translation"] = translation_short
            
            # Refresh table
            self.ocr_table.setData(self.ocr_data)
            
            self.logger.info(f"[OCR_TABLE] Saved changes from detail view for row {row} (Font: {selected_font})")
            
            # Đóng detail view sau khi lưu
            self.detail_container.hide()
            self.current_detail_row = None
    
    def on_ocr_result(self, index: int, texts: list):
        """Handle OCR results"""
        self.ocr_results[index] = texts
        
        # Update display if this is the current image
        if index == self.current_image_index:
            self.update_ocr_display(texts)
    
    def on_image_changed(self, index: int):
        """Handle image navigation"""
        self.current_image_index = index
        self.update_current_image_label(index)
        
        # Đóng detail view khi đổi ảnh
        self.detail_container.hide()
        self.current_detail_row = None
        
        # Update OCR display for current image
        if index in self.ocr_results:
            self.update_ocr_display(self.ocr_results[index])
        else:
            self.logger.warning(f"[OCR_TABLE] No OCR results for index {index} yet")
            self.clear_table()
    
    def update_current_image_label(self, index: int):
        """Update current image label"""
        self.current_image_label.setText(f"Current Image: {index + 1}")
    
    def update_ocr_display(self, texts: list):
        """Update OCR table display"""
        if not texts or not any(text.strip() for text in texts):
            self.clear_table()
            self.logger.warning(f"[OCR_TABLE] No valid text to display")
            return
        
        project_manager = ProjectManager()
        characters = project_manager.get_characters()
        
        # Prepare data for CustomTable
        valid_texts = [text for text in texts if text.strip()]
        table_data = []
        
        self.logger.info(f"[OCR_TABLE] Displaying {len(valid_texts)} texts in manga reading order")
        
        for i, text in enumerate(valid_texts):
            # Truncate text if too long
            original_short = text if len(text) <= OCR_DISPLAY_MAX_LENGTH else text[:OCR_DISPLAY_MAX_LENGTH - 3] + "..."
            
            table_data.append({
                "STT": str(i + 1),
                "Character": "-- Select --",
                "Original Text": original_short,
                "Translation": "",
                "_full_original": text,
                "_full_translation": "",
                "_character_id": None,
                "_font_family": None
            })
        
        self.ocr_table.setData(table_data)
        self.ocr_data = table_data
    
    def clear_table(self):
        """Clear OCR table"""
        self.ocr_table.setData([])
        self.ocr_data = []
    
    def on_save_edit(self):
        """Save changes from edit inputs"""
        if self.current_edit_row is None:
            return
        
        row = self.current_edit_row
        
        # Get text from inputs
        full_original = self.original_input.text()
        full_translation = self.translation_input.text()
        
        # Update data
        if row < len(self.ocr_data):
            self.ocr_data[row]["_full_original"] = full_original
            self.ocr_data[row]["_full_translation"] = full_translation
            
            # Update display (truncated)
            max_len = OCR_DISPLAY_MAX_LENGTH
            original_short = full_original if len(full_original) <= max_len else full_original[:max_len - 3] + "..."
            translation_short = full_translation if len(full_translation) <= max_len else full_translation[:max_len - 3] + "..."
            
            self.ocr_data[row]["Original Text"] = original_short
            self.ocr_data[row]["Translation"] = translation_short
            
            # Refresh table
            self.ocr_table.setData(self.ocr_data)
            
            self.logger.info(f"[OCR_TABLE] Saved changes for row {row}")
    
    def on_character_changed(self, row: int, combo_index: int):
        """Handle character selection change"""
        combo = self.ocr_table.cellWidget(row, 1)
        if combo:
            char_id = combo.currentData()
            char_name = combo.currentText()
            
            if char_id:
                self.logger.info(f"[OCR_TABLE] Row {row}: Character changed to {char_name} (ID: {char_id})")
            else:
                self.logger.info(f"[OCR_TABLE] Row {row}: No character selected")
    
    def get_ocr_table_data(self) -> list:
        """Get table data for saving to project"""
        if not hasattr(self, 'ocr_data'):
            return []
        
        data = []
        for i, row in enumerate(self.ocr_data):
            data.append({
                'index': i,
                'character_id': row.get('_character_id'),
                'font_family': row.get('_font_family'),
                'original_text': row.get('_full_original', ''),
                'translated_text': row.get('_full_translation', '')
            })
        
        return data
    
    def refresh_character_comboboxes(self):
        """Refresh character combo boxes after character list changes"""
        # TODO: Implement refresh logic for character dropdowns in table
        pass
    
    def has_assigned_characters(self) -> bool:
        """
        Kiểm tra xem có row nào đã được assign character chưa
        
        Returns:
            bool: True nếu có ít nhất 1 row có character được set
        """
        if not self.ocr_data:
            return False
        
        for row in self.ocr_data:
            char_id = row.get('_character_id')
            if char_id is not None:
                return True
        
        return False
    
    def clear_all_character_assignments(self):
        """Xóa tất cả character assignments trong table"""
        if not self.ocr_data:
            return
        
        for row in self.ocr_data:
            row['_character_id'] = None
            row['Character'] = '-- Select --'
        
        # Refresh table display
        self.ocr_table.setData(self.ocr_data)
        
        self.logger.info("[OCR_TABLE] Cleared all character assignments")

    def on_ocr_region_result(self, text: str):
        """
        Cập nhật Original Text của row đang focus với kết quả OCR mới
        
        Args:
            text: Kết quả OCR từ region selection
        """
        self.logger.info(f"[OCR_TABLE] Received OCR region result: {text[:50] if text else 'EMPTY'}...")
        
        if self.current_detail_row is None:
            self.logger.warning("[OCR_TABLE] No row is currently focused")
            return
        
        row = self.current_detail_row
        
        if row >= len(self.ocr_data):
            self.logger.error(f"[OCR_TABLE] Invalid row index: {row}")
            return
        
        self.logger.info(f"[OCR_TABLE] Updating row {row} with new OCR result: {text[:50]}...")
        
        # Update full original text
        self.ocr_data[row]["_full_original"] = text
        
        # Update truncated display text
        max_len = OCR_DISPLAY_MAX_LENGTH
        original_short = text if len(text) <= max_len else text[:max_len - 3] + "..."
        self.ocr_data[row]["Original Text"] = original_short
        
        self.logger.info(f"[OCR_TABLE] Data updated - Full: {len(text)} chars, Short: {len(original_short)} chars")
        
        # Refresh table display
        self.ocr_table.setData(self.ocr_data)
        self.logger.info(f"[OCR_TABLE] Table refreshed with new data")
        
        # Update detail view nếu đang mở
        if self.detail_container.isVisible():
            self.detail_original_input.setText(text)
            self.logger.info(f"[OCR_TABLE] Detail view updated with new text")
        
        self.logger.info(f"[OCR_TABLE] Row {row} updated successfully")
        
    def refresh_font_comboboxes(self):
        """Refresh font comboboxes sau khi thay đổi font settings"""
        # Không cần làm gì vì font chỉ được chọn ở detail view
        # Font combo sẽ được refresh tự động khi mở detail view
        self.logger.info("[OCR_TABLE] Font settings updated (will refresh on next detail view open)")