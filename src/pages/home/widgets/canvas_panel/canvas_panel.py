from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                              QFileDialog, QScrollArea, QMessageBox, QPushButton, QSplitter)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QKeyEvent, QImage
import logging
import os
from pathlib import Path
from typing import List, Optional

from widget.common.custom_button import CustomButton
from widget.common.custom_modal import CustomModal
from .interactive_image_label import InteractiveImageLabel
from .image_display_widget import ImageDisplayWidget
from .scaled_label import ScaledLabel
from .canvas_utils import get_toggle_button_style, create_nav_button_style
from ...services.image_loader import ImageLoader
from ...services.segmentation_processor import SegmentationProcessor
from ...services.manga_pipeline_processor import MangaPipelineProcessor


class CanvasPanel(QWidget):
    """Canvas panel với 2 sub-panels để hiển thị ảnh gốc và kết quả"""
    
    # Signals
    folder_selected = Signal(str)
    image_changed = Signal(int)
    segmentation_completed = Signal(int, QImage)
    text_detection_completed = Signal(int, QImage)
    ocr_completed = Signal(int, list)
    panel_visibility_changed = Signal(bool, bool)
    ocr_region_selected = Signal(int, int, int, int, int)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.segmentation_results: dict = {}
        self.visualization_results: dict = {}
        self.text_detection_results: dict = {}
        self.ocr_results: dict = {}
        
        self.image_loader = ImageLoader()
        self.segmentation_processor: Optional[SegmentationProcessor] = None
        self.manga_pipeline_processor: Optional[MangaPipelineProcessor] = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        toggle_bar = QWidget()
        toggle_bar.setStyleSheet("""
            QWidget {
                background-color: var(--sidebar-background);
                border-bottom: 1px solid var(--border);
            }
        """)
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(8, 4, 8, 4)
        toggle_layout.setSpacing(8)
        
        self.toggle_left_btn = QPushButton("◀ Ẩn gốc")
        self.toggle_left_btn.setCheckable(True)
        self.toggle_left_btn.setStyleSheet(get_toggle_button_style())
        self.toggle_left_btn.clicked.connect(self.toggle_left_panel)
        toggle_layout.addWidget(self.toggle_left_btn)
        
        toggle_layout.addStretch()
        
        self.toggle_right_btn = QPushButton("Ẩn kết quả ▶")
        self.toggle_right_btn.setCheckable(True)
        self.toggle_right_btn.setStyleSheet(get_toggle_button_style())
        self.toggle_right_btn.clicked.connect(self.toggle_right_panel)
        toggle_layout.addWidget(self.toggle_right_btn)
        
        nav_bar = QWidget()
        nav_bar.setStyleSheet("""
            QWidget {
                background-color: var(--sidebar-background);
                border-bottom: 1px solid var(--border);
            }
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(8, 4, 8, 4)
        nav_layout.setSpacing(8)
        
        self.nav_buttons = []
        button_configs = [
            ("🖼️", "Open Folder"),
            ("▶️", "Run Pipeline"),
            ("💾", "Save"),
            ("↺", "Undo"),
            ("↻", "Redo"),
            ("🔍", "Zoom In"),
            ("🔎", "Zoom Out"),
            ("⚙️", "Settings"),
        ]
        
        for icon, tooltip in button_configs:
            btn = QPushButton(icon)
            btn.setToolTip(tooltip)
            btn.setFixedSize(32, 32)
            btn.setStyleSheet(create_nav_button_style())
            btn.setEnabled(False)
            self.nav_buttons.append(btn)
            nav_layout.addWidget(btn)
        
        nav_layout.addStretch()
        
        main_layout.addWidget(nav_bar)
        
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: var(--border);
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: var(--primary);
            }
        """)
        
        self.left_panel = ImageDisplayWidget("Original Image")
        self.splitter.addWidget(self.left_panel)
        
        self.right_panel = ImageDisplayWidget("Output Image")
        self.splitter.addWidget(self.right_panel)
        
        self.splitter.setSizes([1, 1])
        
        main_layout.addWidget(self.splitter, 1)
        
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()
    
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
        
        if invalid_files:
            self.show_invalid_files_warning(invalid_files, all_files, folder_path)
        else:
            self.process_valid_files(all_files, folder_path)
    
    def show_invalid_files_warning(self, invalid_files: List[str], 
                                   valid_files: List[str], folder_path: str):
        """Hiển thị cảnh báo file không hợp lệ"""
        modal = CustomModal(
            title="File không hợp lệ",
            size="md",
            parent=self
        )
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)
        
        warning_text = QLabel(
            f"Tìm thấy {len(invalid_files)} file không hợp lệ.\n"
            f"Chỉ chấp nhận: .jpg, .jpeg, .png"
        )
        warning_text.setStyleSheet("color: var(--text-primary); font-size: 14px;")
        warning_text.setWordWrap(True)
        
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
                
        valid_files.sort()
        self.image_loader.load_images(valid_files)
        self.folder_selected.emit(folder_path)
    
    def on_images_loaded(self, image_paths: List[str]):
        """Handle images loaded"""
        self.image_paths = image_paths
        self.current_index = 0
        self.segmentation_results.clear()
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        if self.image_paths:
            self.display_current_image()
        
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
        self.left_panel.set_image(image_path=current_path)
        
        self.logger.info(f"[DISPLAY] Current index: {self.current_index}")
        self.logger.info(f"[DISPLAY] Available results:")
        self.logger.info(f"  - Final results: {list(self.segmentation_results.keys())}")
        self.logger.info(f"  - Visualizations: {list(self.visualization_results.keys())}")
        self.logger.info(f"  - Text detections: {list(self.text_detection_results.keys())}")
        
        if self.current_index in self.segmentation_results:
            self.logger.info(f"[DISPLAY] Showing FINAL RESULT for index {self.current_index}")
            result_data = self.segmentation_results[self.current_index]
            pixmap = QPixmap.fromImage(result_data['image'])
            rectangles = result_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.visualization_results:
            self.logger.info(f"[DISPLAY] Showing VISUALIZATION for index {self.current_index}")
            vis_data = self.visualization_results[self.current_index]
            pixmap = QPixmap.fromImage(vis_data['image'])
            rectangles = vis_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.text_detection_results:
            self.logger.info(f"[DISPLAY] Showing TEXT DETECTION for index {self.current_index}")
            result_qimage = self.text_detection_results[self.current_index]
            pixmap = QPixmap.fromImage(result_qimage)
            self.right_panel.set_image(pixmap=pixmap)
        else:
            self.logger.warning(f"[DISPLAY] Right panel: NO RESULTS available for index {self.current_index}")
            self.right_panel.clear()
        
        self.image_changed.emit(self.current_index)
    
    def previous_image(self):
        """Chuyển về ảnh trước"""
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
        else:
            self.logger.warning(f"[NAVIGATE] Cannot go previous - at first image or no images")
    
    def next_image(self):
        """Chuyển sang ảnh sau"""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
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
        
        if not self.manga_pipeline_processor:
            self.manga_pipeline_processor = MangaPipelineProcessor()
            
            self.manga_pipeline_processor.result_ready.connect(self.on_final_result)
            self.manga_pipeline_processor.visualization_ready.connect(self.on_visualization_result)
            self.manga_pipeline_processor.ocr_result_ready.connect(self.on_ocr_result)
            self.manga_pipeline_processor.progress_updated.connect(self.on_pipeline_progress)
            self.manga_pipeline_processor.error_occurred.connect(self.on_pipeline_error)
            self.manga_pipeline_processor.completed.connect(self.on_pipeline_completed)
        
        self.segmentation_results.clear()
        self.visualization_results.clear()
        self.text_detection_results.clear()
        self.ocr_results.clear()
        
        self.manga_pipeline_processor.process_images(self.image_paths)
    
    def on_segmentation_result(self, index: int, result_image: QImage):
        """Handle segmentation result - KHÔNG DÙNG NỮA"""
        self.logger.warning(f"[RESULT] on_segmentation_result called (deprecated) - index: {index}")
        self.segmentation_results[index] = result_image
        
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.segmentation_completed.emit(index, result_image)
    
    def on_text_detection_result(self, index: int, result_image: QImage):
        """Handle text detection result - KHÔNG DÙNG NỮA"""
        self.logger.warning(f"[RESULT] on_text_detection_result called (deprecated) - index: {index}")
        self.text_detection_results[index] = result_image
        
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap)
        
        self.text_detection_completed.emit(index, result_image)
            
    def on_ocr_result(self, index: int, texts: list):
        """Handle OCR result from original segments"""
        self.ocr_results[index] = texts
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
        
    def on_final_result(self, index: int, result_image: QImage, rectangles: list = None):
        """Handle final result từ pipeline (Step 5 - Final composition)"""
        self.logger.info(f"[FINAL] Received final result for index {index}")
        
        self.segmentation_results[index] = {
            'image': result_image,
            'rectangles': rectangles if rectangles else []
        }
        self.logger.info(f"[FINAL] Stored final result. Total stored: {len(self.segmentation_results)}")
        self.logger.info(f"[FINAL] Rectangles order (right-to-left, top-to-bottom): {[r['id'] for r in (rectangles if rectangles else [])]}")
        
        if index == self.current_index:
            self.logger.info(f"[FINAL] Displaying final result for current image {index}")
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
        else:
            self.logger.info(f"[FINAL] Not current image (current={self.current_index}, received={index})")
        
        self.segmentation_completed.emit(index, result_image)
        
    def on_visualization_result(self, index: int, vis_image: QImage, rectangles: list = None):
        """Handle visualization result (step 2 - với hình chữ nhật đỏ)"""
        self.logger.info(f"[VIS] Received visualization for index {index}")
        
        self.visualization_results[index] = {
            'image': vis_image,
            'rectangles': rectangles if rectangles else []
        }
        self.logger.info(f"[VIS] Stored visualization. Total stored: {len(self.visualization_results)}")
        self.logger.info(f"[VIS] Rectangles order (right-to-left, top-to-bottom): {[r['id'] for r in (rectangles if rectangles else [])]}")
        
        if index == self.current_index:
            self.logger.info(f"[VIS] Displaying visualization for current image {index}")
            pixmap = QPixmap.fromImage(vis_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
        else:
            self.logger.info(f"[VIS] Not current image (current={self.current_index}, received={index})")
            
    def on_pipeline_progress(self, current: int, total: int, step: str):
        """Handle pipeline progress updates"""
        status_text = f"Processing {current}/{total}: {step}"
    
    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline error"""
        self.logger.error(f"Pipeline error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Pipeline", error_msg)
        
    def on_ocr_region_selected(self, x: int, y: int, w: int, h: int):
        """Xử lý khi user chọn vùng OCR"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            return
        
        self.ocr_region_selected.emit(x, y, w, h, self.current_index)
        self.logger.info(f"[OCR] Region selected: x={x}, y={y}, w={w}, h={h} on image {self.current_index}")
        self.left_panel.disable_ocr_mode()
    
    def enable_ocr_selection_mode(self):
        """Bật chế độ chọn vùng OCR"""
        self.left_panel.enable_ocr_mode()
        self.logger.info("[OCR] OCR selection mode enabled")
    
    def disable_ocr_selection_mode(self):
        """Tắt chế độ chọn vùng OCR"""
        self.left_panel.disable_ocr_mode()
        self.logger.info("[OCR] OCR selection mode disabled")
    
    def toggle_left_panel(self):
        """Toggle hiển thị left panel"""
        if self.toggle_left_btn.isChecked():
            self.left_panel.hide()
            self.toggle_left_btn.setText("▶ Hiện gốc")
        else:
            self.left_panel.show()
            self.toggle_left_btn.setText("◀ Ẩn gốc")
        
        self.panel_visibility_changed.emit(
            self.left_panel.isVisible(),
            self.right_panel.isVisible()
        )
    
    def toggle_right_panel(self):
        """Toggle hiển thị right panel"""
        if self.toggle_right_btn.isChecked():
            self.right_panel.hide()
            self.toggle_right_btn.setText("◀ Hiện kết quả")
        else:
            self.right_panel.show()
            self.toggle_right_btn.setText("Ẩn kết quả ▶")
        
        self.panel_visibility_changed.emit(
            self.left_panel.isVisible(),
            self.right_panel.isVisible()
        )