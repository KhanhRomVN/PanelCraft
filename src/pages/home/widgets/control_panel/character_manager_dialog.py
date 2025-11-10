from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                              QMessageBox)
from PySide6.QtCore import Qt
import logging

from widget.common.custom_button import CustomButton
from core.project_manager import ProjectManager


class CharacterManagerDialog(QDialog):
    """Dialog để quản lý characters trong manga project"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project_manager = ProjectManager()
        
        self.setWindowTitle("Quản lý nhân vật")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.setup_ui()
        self.load_characters()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Danh sách nhân vật")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: var(--text-primary);
        """)
        layout.addWidget(title)
        
        # Character list
        self.character_list = QListWidget()
        self.character_list.setStyleSheet("""
            QListWidget {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: var(--primary);
            }
        """)
        layout.addWidget(self.character_list)
        
        # Add character section
        add_layout = QHBoxLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập tên nhân vật...")
        self.name_input.setStyleSheet("""
            QLineEdit {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 8px;
            }
        """)
        add_layout.addWidget(self.name_input)
        
        add_btn = CustomButton(text="Thêm", variant="primary", size="sm")
        add_btn.clicked.connect(self.add_character)
        add_layout.addWidget(add_btn)
        
        layout.addLayout(add_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        delete_btn = CustomButton(text="Xóa nhân vật", variant="danger", size="sm")
        delete_btn.clicked.connect(self.delete_character)
        action_layout.addWidget(delete_btn)
        
        action_layout.addStretch()
        
        close_btn = CustomButton(text="Đóng", variant="secondary", size="sm")
        close_btn.clicked.connect(self.accept)
        action_layout.addWidget(close_btn)
        
        layout.addLayout(action_layout)
    
    def load_characters(self):
        """Load characters từ project"""
        self.character_list.clear()
        
        characters = self.project_manager.get_characters()
        
        for char in characters:
            item = QListWidgetItem(char['name'])
            item.setData(Qt.UserRole, char['id'])
            self.character_list.addItem(item)
    
    def add_character(self):
        """Thêm character mới"""
        name = self.name_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên nhân vật!")
            return
        
        # Check duplicate
        characters = self.project_manager.get_characters()
        if any(c['name'].lower() == name.lower() for c in characters):
            QMessageBox.warning(self, "Cảnh báo", f"Nhân vật '{name}' đã tồn tại!")
            return
        
        # Add to project
        success = self.project_manager.add_character(name)
        
        if success:
            self.load_characters()
            self.name_input.clear()
        else:
            QMessageBox.critical(self, "Lỗi", "Không thể thêm nhân vật!")
    
    def delete_character(self):
        """Xóa character được chọn"""
        current_item = self.character_list.currentItem()
        
        if not current_item:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn nhân vật cần xóa!")
            return
        
        char_name = current_item.text()
        char_id = current_item.data(Qt.UserRole)
        
        # Confirm
        reply = QMessageBox.question(
            self,
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa nhân vật '{char_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.project_manager.delete_character(char_id)
            
            if success:
                self.load_characters()
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể xóa nhân vật!")