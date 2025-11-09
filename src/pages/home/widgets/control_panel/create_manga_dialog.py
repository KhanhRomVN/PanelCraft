from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QMessageBox)
from PySide6.QtCore import Qt
import logging

from widget.common.custom_button import CustomButton
from widget.common.custom_input import CustomInput
from core.project_manager import ProjectManager


class CreateMangaDialog(QDialog):
    """Dialog để tạo manga project mới"""
    
    def __init__(self, parent=None, is_switching=False):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project_manager = ProjectManager()
        self.is_switching = is_switching
        
        self.setWindowTitle("Đổi Manga Project" if is_switching else "Tạo Manga Project")
        self.setMinimumWidth(450)
        self.setMinimumHeight(220)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title_text = "Đổi Manga Project" if self.is_switching else "Tạo Manga Project Mới"
        title = QLabel(title_text)
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            padding-bottom: 8px;
        """)
        layout.addWidget(title)
        
        # Description
        desc_text = "Nhập tên cho manga project của bạn" if not self.is_switching else "Nhập tên cho manga project mới"
        desc = QLabel(desc_text)
        desc.setStyleSheet("""
            font-size: 13px;
            color: var(--text-secondary);
            padding-bottom: 16px;
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Manga Name input
        self.name_input = CustomInput(
            label="Tên Manga *",
            placeholder="Ví dụ: One Piece Chapter 1",
            variant="filled",
            size="md"
        )
        layout.addWidget(self.name_input)
        
        # Spacer
        layout.addStretch()
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        cancel_btn = CustomButton(
            text="Hủy",
            variant="secondary",
            size="md"
        )
        cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(cancel_btn)
        
        create_btn_text = "Đổi Project" if self.is_switching else "Tạo Project"
        create_btn = CustomButton(
            text=create_btn_text,
            variant="primary",
            size="md"
        )
        create_btn.clicked.connect(self.on_create)
        action_layout.addWidget(create_btn)
        
        layout.addLayout(action_layout)
    
    def on_create(self):
        """Xử lý tạo/đổi project"""
        name = self.name_input.text().strip()
        
        # Validate
        if not name:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập tên manga!")
            self.name_input.setFocus()
            return
        
        # Create project
        try:
            success = self.project_manager.create_project(
                name=name,
                author="",
                description=""
            )
            
            if success:
                action_text = "Đã đổi" if self.is_switching else "Đã tạo"
                self.logger.info(f"{action_text} manga project: {name}")
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"{action_text} manga project:\n\n{name}"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self, 
                    "Lỗi", 
                    "Không thể tạo project. Vui lòng thử lại."
                )
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            QMessageBox.critical(
                self, 
                "Lỗi", 
                f"Lỗi khi tạo project:\n{str(e)}"
            )
    
    def get_project_data(self):
        """Lấy thông tin project đã tạo"""
        return {
            'name': self.name_input.text().strip(),
            'author': '',
            'description': ''
        }