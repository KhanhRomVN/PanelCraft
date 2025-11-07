from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QFileDialog, QWidget)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QMovie
import os
from core.model_manager import ModelManager, ModelDownloadThread, ModelInfo
from typing import List, Dict


class ModelDownloadModal(QDialog):
    """Modal để tải models - không thể đóng cho đến khi hoàn tất"""
    
    download_completed = Signal()
    
    def __init__(self, model_manager: ModelManager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.missing_models = []
        self.download_threads = []
        self.completed_count = 0
        self.total_count = 0
        self.is_downloading = False
        
        self.setup_ui()
        self.check_models()
        
        # Không cho phép đóng modal
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setModal(True)
        
    def setup_ui(self):
        """Setup UI cho modal"""
        self.setWindowTitle("Thiết lập Models")
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Cần tải các Models để sử dụng ứng dụng")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: var(--text-primary);
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        self.description_label = QLabel()
        self.description_label.setStyleSheet("""
            font-size: 14px;
            color: var(--text-secondary);
        """)
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)
        
        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: var(--text-primary);
            background-color: var(--card-background);
            padding: 12px;
            border-radius: 6px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.hide()
        layout.addWidget(self.status_label)
        
        # Folder path display
        folder_container = QWidget()
        folder_layout = QVBoxLayout(folder_container)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(8)
        
        folder_label = QLabel("Thư mục lưu models:")
        folder_label.setStyleSheet("font-size: 13px; color: var(--text-secondary);")
        
        self.folder_path_label = QLabel()
        self.folder_path_label.setStyleSheet("""
            font-size: 13px;
            color: var(--text-primary);
            background-color: var(--input-background);
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: 4px;
        """)
        self.folder_path_label.setWordWrap(True)
        
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path_label)
        layout.addWidget(folder_container)
        
        # Spacer
        layout.addStretch()
        
        # Buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(12)
        
        button_layout.addStretch()
        
        # Choose folder button
        self.choose_folder_btn = QPushButton("Choose Folder")
        self.choose_folder_btn.setStyleSheet("""
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
            QPushButton:disabled {
                opacity: 0.5;
            }
        """)
        self.choose_folder_btn.clicked.connect(self.choose_folder)
        button_layout.addWidget(self.choose_folder_btn)
        
        # Download button
        self.download_btn = QPushButton("Tải Models")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: var(--button-bg);
                color: var(--button-text);
                border: 1px solid var(--button-border);
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 500;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: var(--button-bg-hover);
            }
            QPushButton:disabled {
                opacity: 0.5;
                background-color: var(--button-second-bg);
            }
        """)
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)
        button_layout.addWidget(self.download_btn)
        
        layout.addWidget(button_container)
    
    def check_models(self):
        """Kiểm tra models còn thiếu"""
        # Lấy hoặc tạo model path mặc định
        if not self.model_manager.get_model_path():
            default_path = os.path.join(os.getcwd(), "panelcrafter_models")
            self.model_manager.set_model_path(default_path)
        
        # Hiển thị path hiện tại
        current_path = self.model_manager.get_model_path()
        self.folder_path_label.setText(current_path)
        
        # Kiểm tra models thiếu
        missing_grouped = self.model_manager.check_missing_models()
        self.missing_models = []
        for models in missing_grouped.values():
            self.missing_models.extend(models)
        
        self.total_count = len(self.missing_models)
        
        if self.total_count == 0:
            self.description_label.setText("✓ Tất cả models đã sẵn sàng!")
            self.download_btn.setText("Hoàn tất")
            self.download_btn.setEnabled(True)
            self.download_btn.clicked.disconnect()
            self.download_btn.clicked.connect(self.accept)
        else:
            total_size = self.model_manager.get_total_download_size(self.missing_models)
            self.description_label.setText(
                f"Cần tải {self.total_count} models (≈ {total_size:.1f} MB)"
            )
            self.download_btn.setEnabled(True)
    
    def choose_folder(self):
        """Chọn thư mục lưu models"""
        if self.is_downloading:
            return
        
        current_path = self.model_manager.get_model_path() or os.getcwd()
        
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục lưu models",
            current_path
        )
        
        if folder:
            self.model_manager.set_model_path(folder)
            self.folder_path_label.setText(folder)
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(folder, exist_ok=True)
            
            # Kiểm tra lại models
            self.check_models()
    
    def start_download(self):
        """Bắt đầu tải models"""
        if self.total_count == 0:
            self.accept()
            return
        
        # Tạo thư mục nếu chưa có
        model_path = self.model_manager.get_model_path()
        os.makedirs(model_path, exist_ok=True)
        
        # Disable buttons
        self.is_downloading = True
        self.choose_folder_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        
        # Show status
        self.status_label.setText(f"Đang tải... (0/{self.total_count})")
        self.status_label.show()
        
        # Bắt đầu tải từng model
        self.completed_count = 0
        self.download_threads = []
        
        for model in self.missing_models:
            thread = ModelDownloadThread(model, model_path)
            thread.finished.connect(self.on_download_finished)
            self.download_threads.append(thread)
            thread.start()
    
    def on_download_finished(self, filename: str, success: bool, message: str):
        """Xử lý khi tải xong một model"""
        self.completed_count += 1
        
        if success:
            self.status_label.setText(
                f"Đang tải... ({self.completed_count}/{self.total_count})"
            )
        else:
            self.status_label.setText(f"Lỗi tải {filename}: {message}")
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #dc2626;
                background-color: var(--card-background);
                padding: 12px;
                border-radius: 6px;
            """)
            # Re-enable buttons to retry
            self.is_downloading = False
            self.choose_folder_btn.setEnabled(True)
            self.download_btn.setEnabled(True)
            return
        
        # Kiểm tra xem đã tải xong tất cả chưa
        if self.completed_count >= self.total_count:
            self.status_label.setText("✓ Tải xong tất cả models!")
            self.status_label.setStyleSheet("""
                font-size: 14px;
                color: #16a34a;
                background-color: var(--card-background);
                padding: 12px;
                border-radius: 6px;
            """)
            
            # Đợi 1 giây rồi đóng modal
            QTimer.singleShot(1000, self.on_all_completed)
    
    def on_all_completed(self):
        """Xử lý khi tải xong tất cả"""
        self.download_completed.emit()
        self.accept()
    
    def closeEvent(self, event):
        """Ngăn đóng modal khi đang tải"""
        if self.is_downloading:
            event.ignore()
        else:
            event.accept()