from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
import logging

from widget.common.custom_button import CustomButton


class SwitchProjectDialog(QDialog):
    """Dialog cảnh báo khi chuyển đổi project có dữ liệu"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("Cảnh báo")
        self.setMinimumWidth(450)
        self.setMinimumHeight(200)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Warning icon + title
        title_layout = QHBoxLayout()
        
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("""
            font-size: 48px;
            padding-right: 16px;
        """)
        title_layout.addWidget(icon_label)
        
        title = QLabel("Chuyển đổi Manga Project")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
        """)
        title_layout.addWidget(title, 1)
        
        layout.addLayout(title_layout)
        
        # Warning message
        warning_text = QLabel(
            "Bạn đang có dữ liệu OCR với characters đã được gán.\n\n"
            "Nếu chuyển sang project mới, tất cả dữ liệu này sẽ bị xóa:\n"
            "• Characters đã gán cho các text bubble\n"
            "• Font đã chọn\n"
            "• Bản dịch đã nhập (nếu có)\n\n"
            "Bạn có chắc chắn muốn tiếp tục?"
        )
        warning_text.setStyleSheet("""
            color: var(--text-primary);
            font-size: 14px;
            line-height: 1.6;
            padding: 16px;
            background-color: var(--sidebar-background);
            border-radius: 8px;
            border-left: 4px solid #f59e0b;
        """)
        warning_text.setWordWrap(True)
        layout.addWidget(warning_text)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        cancel_btn = CustomButton(
            text="Hủy bỏ",
            variant="secondary",
            size="md"
        )
        cancel_btn.clicked.connect(self.reject)
        action_layout.addWidget(cancel_btn)
        
        confirm_btn = CustomButton(
            text="Xác nhận và xóa dữ liệu",
            variant="danger",
            size="md"
        )
        confirm_btn.clicked.connect(self.accept)
        action_layout.addWidget(confirm_btn)
        
        layout.addLayout(action_layout)