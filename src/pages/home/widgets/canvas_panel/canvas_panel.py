from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                              QFileDialog, QScrollArea, QMessageBox)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QKeyEvent, QImage
import logging
import os
from pathlib import Path
from typing import List, Optional

from widget.common.custom_button import CustomButton
from widget.common.custom_modal import CustomModal
from ...services.image_loader import ImageLoader
from ...services.segmentation_processor import SegmentationProcessor
from ...services.manga_pipeline_processor import MangaPipelineProcessor


class ImageDisplayWidget(QWidget):
    """Widget hiển thị ảnh với scroll"""
    
    def __init__(self, title: str = ""):
        super().__init__()
        self.current_pixmap: Optional[QPixmap] = None
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: var(--text-primary);
                padding: 8px;
                background-color: var(--sidebar-background);
                border-radius: 4px;
            """)
            layout.addWidget(title_label)
        
        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid var(--border);
                border-radius: 4px;
                background-color: var(--card-background);
            }
        """)
        
        # Image label - use ScaledLabel for auto-resize
        self.image_label = ScaledLabel()
        self.image_label.setText("No image loaded")
        
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
    
    def set_image(self, image_path: str = None, pixmap: QPixmap = None):
        """Set image to display"""
        if pixmap:
            self.current_pixmap = pixmap
        elif image_path and os.path.exists(image_path):
            self.current_pixmap = QPixmap(image_path)
        else:
            self.current_pixmap = None
        
        if self.current_pixmap and not self.current_pixmap.isNull():
            # ScaledLabel sẽ tự động scale
            self.image_label.setPixmap(self.current_pixmap)
        else:
            self.image_label.setText("Failed to load image")
    
    def clear(self):
        """Clear image"""
        self.current_pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image loaded")

class ScaledLabel(QLabel):
    """Custom QLabel tự động scale pixmap theo width container"""
    
    def __init__(self):
        super().__init__()
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: var(--card-background);
                padding: 8px;
            }
        """)
    
    def setPixmap(self, pixmap: QPixmap):
        """Override setPixmap để lưu original pixmap"""
        if pixmap:
            self._pixmap = pixmap
            self.updatePixmap()
        else:
            self._pixmap = None
            super().setPixmap(QPixmap())
    
    def resizeEvent(self, event):
        """Auto resize pixmap khi label resize"""
        super().resizeEvent(event)
        if self._pixmap:
            self.updatePixmap()
    
    def updatePixmap(self):
        """Scale pixmap to fit width while keeping aspect ratio"""
        if not self._pixmap or self._pixmap.isNull():
            return
        
        # Scale to fit width of label, keep aspect ratio
        available_width = self.width() - 16  # Trừ padding
        available_height = self.height() - 16
        
        if available_width <= 0 or available_height <= 0:
            return
        
        scaled_pixmap = self._pixmap.scaled(
            available_width,
            available_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        super().setPixmap(scaled_pixmap)

class CanvasPanel(QWidget):
    """Canvas panel với 2 sub-panels để hiển thị ảnh gốc và kết quả"""
    
    # Signals
    folder_selected = Signal(str)
    image_changed = Signal(int)  # Current index
    segmentation_completed = Signal(int, QImage)  # index, result image
    text_detection_completed = Signal(int, QImage)  # Thêm signal mới
    ocr_completed = Signal(int, list)  # Thêm signal mới
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.segmentation_results: dict = {}  # {index: QImage}
        self.text_detection_results: dict = {}  # {index: QImage} - Thêm dictionary mới
        self.ocr_results: dict = {}  # {index: list} - Thêm dictionary mới
        
        self.image_loader = ImageLoader()
        self.segmentation_processor: Optional[SegmentationProcessor] = None
        self.manga_pipeline_processor: Optional[MangaPipelineProcessor] = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Left panel - Original image
        self.left_panel = ImageDisplayWidget("Original Image")
        layout.addWidget(self.left_panel, 1)
        
        # Right panel - Segmentation result
        self.right_panel = ImageDisplayWidget("Processing Results")
        layout.addWidget(self.right_panel, 1)
        
        # Apply focus to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()  # Set focus ngay khi init
    
    def setup_connections(self):
        """Setup signal connections"""
        self.image_loader.images_loaded.connect(self.on_images_loaded)
        self.image_loader.error_occurred.connect(self.on_load_error)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key.Key_Up:
            self.previous_image()
            event.accept()
        elif event.key() == Qt.Key.Key_Down:
            self.next_image()
            event.accept()
        else:
            super().keyPressEvent(event)
            
    def mousePressEvent(self, event):
        """Giữ focus khi click vào panel"""
        super().mousePressEvent(event)
        self.setFocus()
        
    def showEvent(self, event):
        """Set focus khi widget được hiển thị"""
        super().showEvent(event)
        self.setFocus()
    
    def open_folder_dialog(self):
        """Mở dialog chọn folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Chọn thư mục chứa ảnh",
            os.path.expanduser("~")
        )
        
        if folder:
            self.load_folder(folder)
    
    def load_folder(self, folder_path: str):
        """Load folder với validation"""
        
        # Scan folder for valid images
        valid_extensions = ('.jpg', '.jpeg', '.png')
        all_files = []
        invalid_files = []
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                if file.lower().endswith(valid_extensions):
                    all_files.append(file_path)
                else:
                    invalid_files.append(file)
        
        # Show warning if there are invalid files
        if invalid_files:
            self.show_invalid_files_warning(invalid_files, all_files, folder_path)
        else:
            self.process_valid_files(all_files, folder_path)
    
    def show_invalid_files_warning(self, invalid_files: List[str], 
                                   valid_files: List[str], folder_path: str):
        """Hiển thị cảnh báo file không hợp lệ"""
        # Create custom modal
        modal = CustomModal(
            title="File không hợp lệ",
            size="md",
            parent=self
        )
        
        # Create warning content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        
        warning_text = QLabel(
            f"Tìm thấy {len(invalid_files)} file không hợp lệ.\n"
            f"Chỉ chấp nhận: .jpg, .jpeg, .png"
        )
        warning_text.setStyleSheet("color: var(--text-primary); font-size: 14px;")
        warning_text.setWordWrap(True)
        
        # List some invalid files
        if len(invalid_files) <= 5:
            files_text = "\n".join(invalid_files)
        else:
            files_text = "\n".join(invalid_files[:5]) + f"\n... và {len(invalid_files) - 5} file khác"
        
        files_label = QLabel(files_text)
        files_label.setStyleSheet("""
            color: var(--text-secondary);
            font-size: 12px;
            padding: 8px;
            background-color: var(--sidebar-background);
            border-radius: 4px;
        """)
        files_label.setWordWrap(True)
        
        content_layout.addWidget(warning_text)
        content_layout.addWidget(files_label)
        
        modal.setBody(content)
        modal.setFooterVisible(True)
        modal.setActionButton("Continue", "primary")
        modal.cancel_button.setText("Cancel")
        
        # Handle response
        result = modal.exec()
        
        if result == CustomModal.Accepted:
            self.process_valid_files(valid_files, folder_path)
    
    def process_valid_files(self, valid_files: List[str], folder_path: str):
        """Xử lý các file hợp lệ"""
        if not valid_files:
            QMessageBox.warning(
                self,
                "Không có ảnh",
                "Không tìm thấy file ảnh hợp lệ trong thư mục."
            )
            return
                
        # Sort files
        valid_files.sort()
        
        # Load images
        self.image_loader.load_images(valid_files)
        
        # Emit folder selected
        self.folder_selected.emit(folder_path)
    
    def on_images_loaded(self, image_paths: List[str]):
        """Handle images loaded"""
        self.image_paths = image_paths
        self.current_index = 0
        self.segmentation_results.clear()
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        # Display first image
        if self.image_paths:
            self.display_current_image()
        
        # Set focus để nhận keyboard events
        self.setFocus()
    
    def on_load_error(self, error_msg: str):
        """Handle load error"""
        self.logger.error(f"[ERROR] Load error: {error_msg}")
        QMessageBox.critical(self, "Lỗi", error_msg)
    
    def display_current_image(self):
        """Hiển thị ảnh hiện tại với khả năng hiển thị kết quả pipeline"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            self.logger.warning(f"[DISPLAY] Cannot display - no images or invalid index: {self.current_index}")
            return
        
        current_path = self.image_paths[self.current_index]
        
        # Display original image on left panel
        self.left_panel.set_image(image_path=current_path)
        
        # Priority: Segmentation (Step 5 - Final Result) > Text Detection > Nothing
        # Chú ý: segmentation_results chứa kết quả STEP 5 từ pipeline
        if self.current_index in self.segmentation_results:
            result_qimage = self.segmentation_results[self.current_index]
            pixmap = QPixmap.fromImage(result_qimage)
            self.right_panel.set_image(pixmap=pixmap)
        elif self.current_index in self.text_detection_results:
            result_qimage = self.text_detection_results[self.current_index]
            pixmap = QPixmap.fromImage(result_qimage)
            self.right_panel.set_image(pixmap=pixmap)
        else:
            self.right_panel.clear()
            self.logger.warning(f"[DISPLAY] Right panel: NO RESULTS available for index {self.current_index}")
        
        # Emit signal for control panel
        self.image_changed.emit(self.current_index)
    
    def previous_image(self):
        """Chuyển về ảnh trước"""
        if self.image_paths and self.current_index > 0:
            old_index = self.current_index
            self.current_index -= 1
            self.display_current_image()
        else:
            self.logger.warning(f"[NAVIGATE] Cannot go previous - at first image or no images")
    
    def next_image(self):
        """Chuyển sang ảnh sau"""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            old_index = self.current_index
            self.current_index += 1
            self.display_current_image()
        else:
            self.logger.warning(f"[NAVIGATE] Cannot go next - at last image or no images")
    
    def start_segmentation(self):
        """Bắt đầu full manga pipeline (5 steps)"""
        if not self.image_paths:
            self.logger.warning("[PIPELINE] Cannot start - no images loaded")
            QMessageBox.warning(self, "Cảnh báo", "Chưa load ảnh nào.")
            return
        
        # Initialize manga pipeline processor
        if not self.manga_pipeline_processor:
            self.manga_pipeline_processor = MangaPipelineProcessor()
            
            # Connect signals
            self.manga_pipeline_processor.result_ready.connect(self.on_final_result)
            self.manga_pipeline_processor.ocr_result_ready.connect(self.on_ocr_result)  # THÊM DÒNG NÀY
            self.manga_pipeline_processor.progress_updated.connect(self.on_pipeline_progress)
            self.manga_pipeline_processor.error_occurred.connect(self.on_pipeline_error)
            self.manga_pipeline_processor.completed.connect(self.on_pipeline_completed)
        
        # Clear previous results
        self.segmentation_results.clear()
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        # Start processing
        self.manga_pipeline_processor.process_images(self.image_paths)
    
    def on_segmentation_result(self, index: int, result_image: QImage):
        """Handle segmentation result - KHÔNG DÙNG NỮA (chỉ dùng on_final_result)"""
        self.logger.warning(f"[RESULT] on_segmentation_result called (deprecated) - index: {index}")
        self.segmentation_results[index] = result_image
        
        # Update display if this is the current image
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.segmentation_completed.emit(index, result_image)
    
    def on_text_detection_result(self, index: int, result_image: QImage):
        """Handle text detection result - KHÔNG DÙNG NỮA"""
        self.logger.warning(f"[RESULT] on_text_detection_result called (deprecated) - index: {index}")
        self.text_detection_results[index] = result_image
        
        # Update display if this is the current image
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.text_detection_completed.emit(index, result_image)
            
    def on_ocr_result(self, index: int, texts: list):
        """Handle OCR result from original segments"""
        
        # Store result
        self.ocr_results[index] = texts
        
        # Emit signal để ControlPanel có thể hiển thị kết quả OCR
        self.ocr_completed.emit(index, texts)
            
    def on_pipeline_completed(self):
        """Handle pipeline completion"""
        QMessageBox.information(
            self, 
            "Hoàn thành", 
            f"Đã xử lý xong {len(self.image_paths)} ảnh!\n\n"
            "Pipeline gồm 5 bước:\n"
            "1. Segmentation (YOLOv8)\n"
            "2. Tạo blank canvas\n"
            "3. Text detection & removal\n"
            "4. Cập nhật segments\n"
            "5. Paste lại ảnh gốc (Final Result)"
        )
    
    def on_segmentation_error(self, error_msg: str):
        """Handle segmentation error"""
        self.logger.error(f"[ERROR] Segmentation error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Segmentation", error_msg)
        
    def on_final_result(self, index: int, result_image: QImage):
        """Handle final result từ pipeline (Step 5 - Final composition)"""
        
        # Lưu vào segmentation_results để có thể navigate
        self.segmentation_results[index] = result_image
        
        # Update display nếu đây là ảnh hiện tại
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        # Emit signal
        self.segmentation_completed.emit(index, result_image)
            
    def on_pipeline_progress(self, current: int, total: int, step: str):
        """Handle pipeline progress updates"""
        
        # Cập nhật status text nếu có status bar
        status_text = f"Processing {current}/{total}: {step}"
    
    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline error"""
        self.logger.error(f"Pipeline error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Pipeline", error_msg)