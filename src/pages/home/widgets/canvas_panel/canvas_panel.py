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
from ...services.image_loader import ImageLoader
from ...services.segmentation_processor import SegmentationProcessor
from ...services.manga_pipeline_processor import MangaPipelineProcessor
from ...constants.constants import VALID_IMAGE_EXTENSIONS


class ImageDisplayWidget(QWidget):
    """Widget hiển thị ảnh với scroll và rectangles có thể drag"""
    
    def __init__(self, title: str = ""):
        super().__init__()
        self.current_pixmap: Optional[QPixmap] = None
        self.rectangles: List[dict] = []
        self.ocr_mode_enabled: bool = False  # CHẾ ĐỘ OCR DRAG
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
        
        # Image label - use InteractiveImageLabel for drag-drop rectangles
        self.image_label = InteractiveImageLabel()
        self.image_label.setText("No image loaded")
        
        # THÊM: Connect OCR region selected signal
        self.image_label.ocr_region_selected.connect(self.on_ocr_region_selected_internal)
        
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)
    
    def set_image(self, image_path: str = None, pixmap: QPixmap = None, rectangles: List[dict] = None):
        """Set image to display với optional rectangles"""
        if pixmap:
            self.current_pixmap = pixmap
        elif image_path and os.path.exists(image_path):
            self.current_pixmap = QPixmap(image_path)
        else:
            self.current_pixmap = None
        
        # Cập nhật rectangles
        if rectangles is not None:
            self.rectangles = rectangles
            self.image_label.set_rectangles(rectangles)
        
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.image_label.setPixmap(self.current_pixmap)
        else:
            self.image_label.setText("Failed to load image")
    
    def clear(self):
        """Clear image"""
        self.current_pixmap = None
        self.rectangles.clear()
        self.image_label.clear()
        self.image_label.set_rectangles([])
        self.image_label.setText("No image loaded")
    
    def get_rectangles(self) -> List[dict]:
        """Lấy danh sách rectangles hiện tại (sau khi drag)"""
        return self.image_label.get_rectangles()
    
    def on_ocr_region_selected_internal(self, x: int, y: int, w: int, h: int):
        """Handle khi user chọn vùng OCR từ InteractiveImageLabel"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Tìm CanvasPanel (parent của parent vì ImageDisplayWidget nằm trong QSplitter)
        canvas_panel = None
        parent = self.parent()
        
        # Duyệt lên để tìm CanvasPanel
        while parent is not None:
            if hasattr(parent, 'on_ocr_region_selected') and parent.__class__.__name__ == 'CanvasPanel':
                canvas_panel = parent
                break
            parent = parent.parent()
        
        if canvas_panel:
            canvas_panel.on_ocr_region_selected(x, y, w, h)
        else:
            logger.error(f"[IMAGE_DISPLAY] Could not find CanvasPanel in parent hierarchy")
    
    def enable_ocr_mode(self):
        """Bật chế độ OCR drag"""
        self.ocr_mode_enabled = True
        self.image_label.enable_ocr_mode()
    
    def disable_ocr_mode(self):
        """Tắt chế độ OCR drag"""
        self.ocr_mode_enabled = False
        self.image_label.disable_ocr_mode()

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
    image_changed = Signal(int)
    segmentation_completed = Signal(int, QImage)
    text_detection_completed = Signal(int, QImage)
    ocr_completed = Signal(int, list)
    panel_visibility_changed = Signal(bool, bool)
    ocr_region_selected = Signal(int, int, int, int, int)  # x, y, w, h, image_index
    ocr_region_result_ready = Signal(str)  # THÊM: Signal trả kết quả OCR của vùng được chọn
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.segmentation_results: dict = {}  # {index: QImage}
        self.visualization_results: dict = {}  # THÊM: {index: QImage} - Step 2 visualization
        self.text_detection_results: dict = {}  # {index: QImage}
        self.ocr_results: dict = {}  # {index: list}
        
        self.image_loader = ImageLoader()
        self.segmentation_processor: Optional[SegmentationProcessor] = None
        self.manga_pipeline_processor: Optional[MangaPipelineProcessor] = None
        self.ocr_model = None  # THÊM: Cache OCR model để không load lại nhiều lần
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Toggle buttons bar
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
        self.toggle_left_btn.setStyleSheet(self._get_toggle_button_style())
        self.toggle_left_btn.clicked.connect(self.toggle_left_panel)
        toggle_layout.addWidget(self.toggle_left_btn)
        
        toggle_layout.addStretch()
        
        self.toggle_right_btn = QPushButton("Ẩn kết quả ▶")
        self.toggle_right_btn.setCheckable(True)
        self.toggle_right_btn.setStyleSheet(self._get_toggle_button_style())
        self.toggle_right_btn.clicked.connect(self.toggle_right_panel)
        toggle_layout.addWidget(self.toggle_right_btn)
        
        # Navigation bar với các button icon
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
        
        # Tạo các button với icon (tạm thời chưa có chức năng)
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
            btn.setStyleSheet("""
                QPushButton {
                    background-color: var(--card-background);
                    color: var(--text-primary);
                    border: 1px solid var(--border);
                    border-radius: 4px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: var(--sidebar-background);
                    border-color: var(--primary);
                }
                QPushButton:pressed {
                    background-color: var(--primary);
                }
            """)
            btn.setEnabled(False)  # Tạm thời disable vì chưa có chức năng
            self.nav_buttons.append(btn)
            nav_layout.addWidget(btn)
        
        nav_layout.addStretch()
        
        main_layout.addWidget(nav_bar)
        
        # Splitter chứa 2 panels
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
        
        # Left panel - Original image
        self.left_panel = ImageDisplayWidget("Original Image")
        self.splitter.addWidget(self.left_panel)
        
        # Right panel
        self.right_panel = ImageDisplayWidget("Output Image")
        self.splitter.addWidget(self.right_panel)
        
        # Set initial sizes (50-50)
        self.splitter.setSizes([1, 1])
        
        main_layout.addWidget(self.splitter, 1)
        
        # Apply focus to receive keyboard events
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
        valid_extensions = VALID_IMAGE_EXTENSIONS
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
        
        # Priority: Final Result (Step 5) > Visualization (Step 2) > Text Detection > Nothing
        if self.current_index in self.segmentation_results:
            result_data = self.segmentation_results[self.current_index]
            pixmap = QPixmap.fromImage(result_data['image'])
            rectangles = result_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.visualization_results:
            # Hiển thị visualization với hình chữ nhật đỏ (Step 2)
            vis_data = self.visualization_results[self.current_index]
            pixmap = QPixmap.fromImage(vis_data['image'])
            rectangles = vis_data.get('rectangles', [])
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles)
        elif self.current_index in self.text_detection_results:
            result_qimage = self.text_detection_results[self.current_index]
            pixmap = QPixmap.fromImage(result_qimage)
            self.right_panel.set_image(pixmap=pixmap)
        else:
            self.logger.warning(f"[DISPLAY] Right panel: NO RESULTS available for index {self.current_index}")
            self.right_panel.clear()
        
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
            self.manga_pipeline_processor.visualization_ready.connect(self.on_visualization_result)  # THÊM
            self.manga_pipeline_processor.ocr_result_ready.connect(self.on_ocr_result)
            self.manga_pipeline_processor.progress_updated.connect(self.on_pipeline_progress)
            self.manga_pipeline_processor.error_occurred.connect(self.on_pipeline_error)
            self.manga_pipeline_processor.completed.connect(self.on_pipeline_completed)
        
        # Clear previous results
        self.segmentation_results.clear()
        self.visualization_results.clear()  # THÊM
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
        
    def on_final_result(self, index: int, result_image: QImage, rectangles: list = None):
        """Handle final result từ pipeline (Step 5 - Final composition)"""        
        # Rectangles đã được sắp xếp trong pipeline theo thứ tự đọc manga
        # Lưu vào segmentation_results với rectangles metadata
        self.segmentation_results[index] = {
            'image': result_image,
            'rectangles': rectangles if rectangles else []
        }
        
        # Update display nếu đây là ảnh hiện tại
        if index == self.current_index:
            pixmap = QPixmap.fromImage(result_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
        
        # Emit signal
        self.segmentation_completed.emit(index, result_image)
        
    def on_visualization_result(self, index: int, vis_image: QImage, rectangles: list = None):
        """Handle visualization result (step 2 - với hình chữ nhật đỏ)"""        
        # Rectangles đã được sắp xếp trong pipeline theo thứ tự đọc manga
        # Lưu vào dictionary với rectangles metadata
        self.visualization_results[index] = {
            'image': vis_image,
            'rectangles': rectangles if rectangles else []
        }
        
        # Update display nếu đây là ảnh hiện tại
        if index == self.current_index:
            pixmap = QPixmap.fromImage(vis_image)
            self.right_panel.set_image(pixmap=pixmap, rectangles=rectangles if rectangles else [])
            
    def on_pipeline_progress(self, current: int, total: int, step: str):
        """Handle pipeline progress updates"""
        
        # Cập nhật status text nếu có status bar
        status_text = f"Processing {current}/{total}: {step}"
    
    def on_pipeline_error(self, error_msg: str):
        """Handle pipeline error"""
        self.logger.error(f"Pipeline error: {error_msg}")
        QMessageBox.critical(self, "Lỗi Pipeline", error_msg)
        
    def on_ocr_region_selected(self, x: int, y: int, w: int, h: int):
        """Xử lý khi user chọn vùng OCR"""
        if not self.image_paths or self.current_index >= len(self.image_paths):
            self.logger.error(f"[OCR] Cannot process - no images or invalid index")
            return
                
        # 1. Crop image theo coordinates
        import cv2
        current_image_path = self.image_paths[self.current_index]
        image = cv2.imread(current_image_path)
        
        if image is None:
            self.logger.error(f"[OCR] Failed to load image: {current_image_path}")
            self.left_panel.disable_ocr_mode()
            return
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop region
        cropped = image_rgb[y:y+h, x:x+w]
        
        if cropped.size == 0:
            self.logger.error(f"[OCR] Invalid crop region")
            self.left_panel.disable_ocr_mode()
            return
        
        # 2. Chạy OCR model
        ocr_text = self._run_ocr_on_region(cropped)
        
        # 3. Emit signal với kết quả OCR
        self.ocr_region_result_ready.emit(ocr_text)
                
        # Tắt OCR mode sau khi chọn xong
        self.left_panel.disable_ocr_mode()
    
    def _run_ocr_on_region(self, cropped_image_rgb: 'np.ndarray') -> str:
        """
        Chạy OCR trên vùng ảnh đã crop
        
        Args:
            cropped_image_rgb: Ảnh RGB đã crop (numpy array)
        
        Returns:
            str: Kết quả OCR
        """
        try:
            # Load OCR model nếu chưa có (cache để tránh load lại nhiều lần)
            if self.ocr_model is None:
                from manga_ocr import MangaOcr
                self.ocr_model = MangaOcr()
            
            # Convert numpy array to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(cropped_image_rgb)
            
            # Run OCR
            text = self.ocr_model(pil_image)
            
            return text
            
        except Exception as e:
            self.logger.error(f"[OCR] Error running OCR: {e}")
            import traceback
            traceback.print_exc()
            return "[OCR ERROR]"
    
    def enable_ocr_selection_mode(self):
        """Bật chế độ chọn vùng OCR"""
        self.left_panel.enable_ocr_mode()
    
    def disable_ocr_selection_mode(self):
        """Tắt chế độ chọn vùng OCR"""
        self.left_panel.disable_ocr_mode()
    
    def toggle_left_panel(self):
        """Toggle hiển thị left panel"""
        if self.toggle_left_btn.isChecked():
            self.left_panel.hide()
            self.toggle_left_btn.setText("▶ Hiện gốc")
        else:
            self.left_panel.show()
            self.toggle_left_btn.setText("◀ Ẩn gốc")
        
        # Emit signal để HomePage điều chỉnh layout
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
        
        # Emit signal để HomePage điều chỉnh layout
        self.panel_visibility_changed.emit(
            self.left_panel.isVisible(),
            self.right_panel.isVisible()
        )
    
    def _get_toggle_button_style(self) -> str:
        """Style cho toggle buttons"""
        return """
            QPushButton {
                background-color: var(--card-background);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: var(--sidebar-background);
                border-color: var(--primary);
            }
            QPushButton:checked {
                background-color: var(--primary);
                color: white;
                border-color: var(--primary);
            }
        """