from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QListWidget, QListWidgetItem, QMessageBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
import logging

from widget.common.custom_button import CustomButton
from core.project_manager import ProjectManager


class SelectProjectDialog(QDialog):
    """Dialog để chọn/đổi manga project từ danh sách có sẵn"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.project_manager = ProjectManager()
        self.selected_project_id = None
        
        self.setWindowTitle("Đổi Manga Project")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)
        
        self.setup_ui()
        self.load_projects()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Title
        title = QLabel("Chọn Manga Project")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
            padding-bottom: 8px;
        """)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Chọn một manga project từ danh sách bên dưới để chuyển đổi")
        desc.setStyleSheet("""
            font-size: 13px;
            color: var(--text-secondary);
            padding-bottom: 16px;
        """)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Project list
        self.project_list = QListWidget()
        self.project_list.setStyleSheet("""
            QListWidget {
                background-color: var(--input-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 4px;
            }
            QListWidget::item:hover {
                background-color: var(--sidebar-background);
            }
            QListWidget::item:selected {
                background-color: var(--primary);
                color: white;
            }
        """)
        self.project_list.itemDoubleClicked.connect(self.on_project_double_clicked)
        layout.addWidget(self.project_list, 1)
        
        # Buttons row
        button_row = QHBoxLayout()
        
        # Create new project button
        create_new_btn = CustomButton(
            text="➕ Tạo Project Mới",
            variant="secondary",
            size="md"
        )
        create_new_btn.clicked.connect(self.on_create_new)
        button_row.addWidget(create_new_btn)
        
        button_row.addStretch()
        
        # Cancel button
        cancel_btn = CustomButton(
            text="Hủy",
            variant="secondary",
            size="md"
        )
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        
        # Select button
        select_btn = CustomButton(
            text="Chọn Project",
            variant="primary",
            size="md"
        )
        select_btn.clicked.connect(self.on_select)
        button_row.addWidget(select_btn)
        
        layout.addLayout(button_row)
    
    def load_projects(self):
        """Load danh sách manga projects"""
        self.project_list.clear()
        
        # Lấy danh sách projects từ ProjectManager
        try:
            projects = self.project_manager.get_all_projects()
        except AttributeError:
            # Fallback: Nếu method chưa có, chỉ hiển thị current project
            self.logger.warning("ProjectManager.get_all_projects() not implemented yet")
            current_project_info = self.project_manager.get_current_project_info()
            
            if current_project_info:
                projects = [current_project_info]
            else:
                projects = []
        
        if not projects:
            # Hiển thị thông báo nếu chưa có project nào
            empty_item = QListWidgetItem("Chưa có manga project nào. Hãy tạo mới!")
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)  # Disable selection
            empty_item.setForeground(Qt.gray)
            self.project_list.addItem(empty_item)
            return
        
        # Lấy current project để highlight
        current_project_info = self.project_manager.get_current_project_info()
        current_project_id = current_project_info.get('id') if current_project_info else None
        
        for project in projects:
            project_id = project.get('id')
            project_name = project.get('name', 'Unknown')
            project_author = project.get('author', '')
            
            # Format display text
            display_text = f"📖 {project_name}"
            if project_author:
                display_text += f"\n   ✍️ {project_author}"
            
            # Add indicator if this is current project
            if project_id == current_project_id:
                display_text += "\n   ✓ (Đang sử dụng)"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, project_id)
            
            # Chỉ mark current project bằng visual indicator, KHÔNG auto select
            if project_id == current_project_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                # Sử dụng màu primary
                item.setForeground(QBrush(QColor("#3b82f6")))
            
            self.project_list.addItem(item)
    
    def on_project_double_clicked(self, item: QListWidgetItem):
        """Handle double-click để chọn nhanh"""
        self.on_select()
    
    def on_select(self):
        """Xử lý chọn project"""
        current_item = self.project_list.currentItem()
        
        if not current_item:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn một manga project!")
            return
        
        project_id = current_item.data(Qt.UserRole)
        project_name = current_item.text().split('\n')[0].replace('📖 ', '')
        
        if not project_id:
            return
        
        # Check if selecting current project
        current_project_info = self.project_manager.get_current_project_info()
        current_project_id = current_project_info.get('id') if current_project_info else None
        
        if project_id == current_project_id:
            QMessageBox.information(
                self,
                "Thông báo",
                "Bạn đang sử dụng project này rồi!"
            )
            return
        
        # Switch to selected project
        try:
            success = self.project_manager.switch_project(project_id)
        except AttributeError:
            # Fallback: Nếu method chưa có
            self.logger.error("ProjectManager.switch_project() not implemented yet")
            QMessageBox.critical(
                self,
                "Lỗi",
                "Chức năng đổi project chưa được triển khai.\nVui lòng liên hệ developer."
            )
            return
        
        if success:
            self.selected_project_id = project_id
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Lỗi",
                "Không thể chuyển đổi project. Vui lòng thử lại."
            )
    
    def on_create_new(self):
        """Mở dialog tạo project mới"""
        from .create_manga_dialog import CreateMangaDialog
        
        dialog = CreateMangaDialog(self, is_switching=False)
        if dialog.exec():
            self.load_projects()
    
    def get_selected_project_id(self):
        """Lấy ID của project đã chọn"""
        return self.selected_project_id