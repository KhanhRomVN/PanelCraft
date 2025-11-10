from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                              QMessageBox, QCheckBox, QRadioButton, QButtonGroup, QWidget)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontDatabase, QFont
import logging

from widget.common.custom_button import CustomButton
from widget.common.custom_input import CustomInput
from core.font_manager import FontManager


class FontManagerDialog(QDialog):
    """Dialog để quản lý fonts hiển thị và font mặc định"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.font_manager = FontManager()
        
        self.setWindowTitle("Quản lý Font")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        # Load system fonts
        font_db = QFontDatabase()
        self.all_fonts = sorted(font_db.families())
        
        # Setup debounce timer cho search
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.setInterval(300)  # 300ms delay
        self.search_timer.timeout.connect(self.perform_search)
        self.pending_search_text = ""
        
        self.setup_ui()
        self.load_font_settings()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Title
        title = QLabel("Quản lý Font")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            padding-bottom: 8px;
        """)
        layout.addWidget(title)
        
        # Description
        desc = QLabel(
            "Chọn các fonts sẽ hiển thị trong dropdown và font mặc định.\n"
            "Font mặc định sẽ tự động áp dụng cho text bubbles mới."
        )
        desc.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            padding-bottom: 12px;
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Search box
        self.search_input = CustomInput(
            placeholder="Tìm kiếm font...",
            variant="filled",
            size="sm"
        )
        self.search_input.textChanged.connect(self.on_search_changed)
        layout.addWidget(self.search_input)
        
        # Font list với checkboxes
        list_label = QLabel("Danh sách Fonts (Tick để thêm):")
        list_label.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: var(--text-primary);
            padding-top: 8px;
        """)
        layout.addWidget(list_label)
        
        self.font_list = QListWidget()
        self.font_list.setStyleSheet("""
            QListWidget {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: var(--sidebar-background);
            }
        """)
        layout.addWidget(self.font_list)
        
        # Populate font list
        self.populate_font_list()
        
        # Selected fonts label
        selected_label = QLabel("Fonts đã chọn:")
        selected_label.setStyleSheet("""
            font-size: 13px;
            font-weight: bold;
            color: var(--text-primary);
            padding-top: 8px;
        """)
        layout.addWidget(selected_label)
        
        self.selected_font_list = QListWidget()
        self.selected_font_list.setStyleSheet("""
            QListWidget {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 4px;
                max-height: 200px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: var(--sidebar-background);
            }
        """)
        layout.addWidget(self.selected_font_list)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        select_all_btn = CustomButton(text="Chọn tất cả", variant="secondary", size="sm")
        select_all_btn.clicked.connect(self.select_all_fonts)
        action_layout.addWidget(select_all_btn)
        
        deselect_all_btn = CustomButton(text="Bỏ chọn tất cả", variant="secondary", size="sm")
        deselect_all_btn.clicked.connect(self.deselect_all_fonts)
        action_layout.addWidget(deselect_all_btn)
        
        action_layout.addStretch()
        
        cancel_btn = CustomButton(text="Hủy", variant="secondary", size="sm")
        cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(cancel_btn)
        
        save_btn = CustomButton(text="Lưu", variant="primary", size="sm")
        save_btn.clicked.connect(self.save_settings)
        action_layout.addWidget(save_btn)
        
        layout.addLayout(action_layout)
    
    def populate_font_list(self, filter_text: str = ""):
        """Populate font list với checkboxes - Optimized"""
        # Block signals để tránh trigger nhiều events
        self.font_list.blockSignals(True)
        self.font_list.setUpdatesEnabled(False)
        
        self.font_list.clear()
        
        # Filter fonts trước khi tạo widgets
        filtered_fonts = self.all_fonts
        if filter_text:
            filter_lower = filter_text.lower()
            filtered_fonts = [f for f in self.all_fonts if filter_lower in f.lower()]
        
        # Giới hạn số lượng fonts hiển thị để tránh lag
        max_display = 500
        if len(filtered_fonts) > max_display:
            filtered_fonts = filtered_fonts[:max_display]
        
        for font_family in filtered_fonts:
            item = QListWidgetItem()
            self.font_list.addItem(item)
            
            # Create checkbox widget
            checkbox = QCheckBox(font_family)
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: var(--text-primary);
                    font-size: 13px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                }
            """)
            
            # Set font preview
            font = QFont(font_family)
            font.setPointSize(11)
            checkbox.setFont(font)
            
            # Connect checkbox để tự động thêm/xóa font khi tick
            checkbox.stateChanged.connect(
                lambda state, fname=font_family: self.on_font_checkbox_changed(fname, state)
            )
            
            self.font_list.setItemWidget(item, checkbox)
        
        # Re-enable updates và signals
        self.font_list.setUpdatesEnabled(True)
        self.font_list.blockSignals(False)
    
    def on_search_changed(self, text: str):
        """Handle search input change với debounce"""
        self.pending_search_text = text
        self.search_timer.stop()
        self.search_timer.start()
    
    def perform_search(self):
        """Thực hiện search sau delay (debounced)"""
        self.populate_font_list(self.pending_search_text)
        self.load_font_settings()  # Re-apply selections after filter
    
    def on_font_checkbox_changed(self, font_family: str, state: int):
        """Xử lý khi tick/untick checkbox font"""
        from PySide6.QtCore import Qt
        
        if state == Qt.CheckState.Checked.value:
            # Thêm font vào selected list
            self.add_font_to_selected(font_family)
        else:
            # Xóa font khỏi selected list
            self.remove_font_from_selected(font_family)
            
    def add_font_to_selected(self, font_family: str):
        """Thêm font vào danh sách đã chọn"""
        # Kiểm tra xem font đã tồn tại chưa
        for i in range(self.selected_font_list.count()):
            item = self.selected_font_list.item(i)
            widget = self.selected_font_list.itemWidget(item)
            if widget and hasattr(widget, 'font_name'):
                if widget.font_name == font_family:
                    return  # Font đã tồn tại
        
        # Tạo custom widget cho selected font item
        item = QListWidgetItem()
        item_widget = self.create_selected_font_widget(font_family)
        
        item.setSizeHint(item_widget.sizeHint())
        self.selected_font_list.addItem(item)
        self.selected_font_list.setItemWidget(item, item_widget)
    
    def remove_font_from_selected(self, font_family: str):
        """Xóa font khỏi danh sách đã chọn"""
        # Tìm và xóa item có font_family tương ứng
        for i in range(self.selected_font_list.count()):
            item = self.selected_font_list.item(i)
            widget = self.selected_font_list.itemWidget(item)
            
            if widget and hasattr(widget, 'font_name'):
                if widget.font_name == font_family:
                    # Xóa item khỏi selected list
                    self.selected_font_list.takeItem(i)
                    # Uncheck checkbox trong danh sách chính
                    self.uncheck_font_in_list(font_family)
                    break
        
    def create_selected_font_widget(self, font_family: str) -> QWidget:
        """Tạo widget cho font item trong selected list với trash và heart icons"""
        from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
        
        container = QWidget()
        container.font_name = font_family  # Store font name for reference
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Font name label với preview
        font_label = QLabel(font_family)
        font = QFont(font_family)
        font.setPointSize(11)
        font_label.setFont(font)
        font_label.setStyleSheet("color: var(--text-primary);")
        layout.addWidget(font_label, 1)
        
        # Heart button (set default)
        heart_btn = QPushButton("❤️")
        heart_btn.setFixedSize(32, 32)
        heart_btn.setToolTip("Đặt làm font mặc định")
        heart_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid var(--border);
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: var(--sidebar-background);
                border-color: var(--primary);
            }
        """)
        heart_btn.clicked.connect(lambda: self.set_default_font(font_family))
        layout.addWidget(heart_btn)
        
        # Trash button (remove)
        trash_btn = QPushButton("🗑️")
        trash_btn.setFixedSize(32, 32)
        trash_btn.setToolTip("Xóa khỏi danh sách")
        trash_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid var(--border);
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: var(--sidebar-background);
                border-color: #ef4444;
            }
        """)
        trash_btn.clicked.connect(lambda: self.remove_font_from_selected(font_family))
        layout.addWidget(trash_btn)
        
        return container
            
    def uncheck_font_in_list(self, font_family: str):
        """Uncheck checkbox của font trong danh sách chính"""
        for i in range(self.font_list.count()):
            item = self.font_list.item(i)
            checkbox = self.font_list.itemWidget(item)
            if checkbox and checkbox.text() == font_family:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
                break
    
    def set_default_font(self, font_family: str):
        """Đặt font làm mặc định"""
        self.font_manager.set_default_font(font_family)
        
        # Update heart icons cho tất cả items
        self.update_heart_icons(font_family)
                
    def update_heart_icons(self, default_font: str):
        """Cập nhật heart icons - highlight font mặc định"""
        for i in range(self.selected_font_list.count()):
            item = self.selected_font_list.item(i)
            widget = self.selected_font_list.itemWidget(item)
            
            if widget and hasattr(widget, 'font_name'):
                # Tìm heart button trong widget
                layout = widget.layout()
                if layout and layout.count() >= 2:
                    heart_btn = layout.itemAt(1).widget()
                    
                    if widget.font_name == default_font:
                        # Đây là font mặc định - highlight
                        heart_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #fef3c7;
                                border: 2px solid #f59e0b;
                                border-radius: 4px;
                                font-size: 16px;
                            }
                            QPushButton:hover {
                                background-color: #fde68a;
                            }
                        """)
                    else:
                        # Font thường
                        heart_btn.setStyleSheet("""
                            QPushButton {
                                background-color: transparent;
                                border: 1px solid var(--border);
                                border-radius: 4px;
                                font-size: 16px;
                            }
                            QPushButton:hover {
                                background-color: var(--sidebar-background);
                                border-color: var(--primary);
                            }
                        """)
    
    
    
    def load_font_settings(self):
        """Load font settings từ FontManager"""
        visible_fonts = self.font_manager.get_visible_fonts()
        default_font = self.font_manager.get_default_font()
        
        # Block signals để tránh trigger events
        self.font_list.blockSignals(True)
        self.selected_font_list.blockSignals(True)
        
        # Set checkboxes
        for i in range(self.font_list.count()):
            item = self.font_list.item(i)
            checkbox = self.font_list.itemWidget(item)
            
            if checkbox:
                checkbox.blockSignals(True)
                checkbox.setChecked(checkbox.text() in visible_fonts)
                checkbox.blockSignals(False)
        
        # Clear và populate selected font list
        self.selected_font_list.clear()
        for font_family in visible_fonts:
            item = QListWidgetItem()
            item_widget = self.create_selected_font_widget(font_family)
            item.setSizeHint(item_widget.sizeHint())
            self.selected_font_list.addItem(item)
            self.selected_font_list.setItemWidget(item, item_widget)
        
        # Update heart icons cho default font
        if default_font:
            self.update_heart_icons(default_font)
        
        # Unblock signals
        self.font_list.blockSignals(False)
        self.selected_font_list.blockSignals(False)
    
    def select_all_fonts(self):
        """Chọn tất cả fonts"""
        for i in range(self.font_list.count()):
            item = self.font_list.item(i)
            checkbox = self.font_list.itemWidget(item)
            if checkbox and not checkbox.isChecked():
                checkbox.setChecked(True)
    
    def deselect_all_fonts(self):
        """Bỏ chọn tất cả fonts"""
        for i in range(self.font_list.count()):
            item = self.font_list.item(i)
            checkbox = self.font_list.itemWidget(item)
            if checkbox and checkbox.isChecked():
                checkbox.setChecked(False)
    
    def save_settings(self):
        """Lưu font settings"""
        # Get selected fonts từ selected_font_list
        selected_fonts = []
        for i in range(self.selected_font_list.count()):
            item = self.selected_font_list.item(i)
            widget = self.selected_font_list.itemWidget(item)
            if widget and hasattr(widget, 'font_name'):
                selected_fonts.append(widget.font_name)
        
        if not selected_fonts:
            QMessageBox.warning(
                self,
                "Cảnh báo",
                "Vui lòng chọn ít nhất một font!"
            )
            return
        
        # Get default font từ font_manager (đã được set qua heart button)
        default_font = self.font_manager.get_default_font()
        
        # Save visible fonts to FontManager
        success = self.font_manager.set_visible_fonts(selected_fonts)
        
        if success:
            default_text = default_font if default_font else "Default (System)"
            QMessageBox.information(
                self,
                "Thành công",
                f"Đã lưu cài đặt font:\n"
                f"• {len(selected_fonts)} fonts hiển thị\n"
                f"• Font mặc định: {default_text}"
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Lỗi",
                "Không thể lưu cài đặt font!"
            )