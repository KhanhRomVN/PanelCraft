import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from PySide6.QtCore import QObject, Signal, QThread
import requests
from tqdm import tqdm
import logging


@dataclass
class ModelInfo:
    """Thông tin về một model"""
    name: str
    url: str
    filename: str
    category: str  # 'text_detection', 'ocr', 'segmentation'
    required: bool = True
    size_mb: Optional[float] = None
    
    def __post_init__(self):
        """Extract filename from URL if not provided"""
        if not self.filename and self.url:
            self.filename = self.url.split('/')[-1]


class ModelDownloadThread(QThread):
    """Thread để tải model không block UI"""
    
    progress = Signal(str, int, int)  # filename, current, total
    finished = Signal(str, bool, str)  # filename, success, message
    
    def __init__(self, model_info: ModelInfo, save_path: str):
        super().__init__()
        self.model_info = model_info
        self.save_path = save_path
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Tải file từ URL"""
        try:
            self.logger.info(f"Downloading {self.model_info.name} from {self.model_info.url}")
            
            response = requests.get(self.model_info.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            file_path = os.path.join(self.save_path, self.model_info.filename)
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(
                            self.model_info.filename,
                            downloaded,
                            total_size
                        )
            
            self.finished.emit(
                self.model_info.filename,
                True,
                f"Tải thành công {self.model_info.name}"
            )
            
        except Exception as e:
            self.logger.error(f"Error downloading {self.model_info.name}: {e}")
            self.finished.emit(
                self.model_info.filename,
                False,
                f"Lỗi: {str(e)}"
            )


class ModelManager(QObject):
    """Quản lý việc kiểm tra và tải models"""
    
    # Danh sách models cần thiết
    REQUIRED_MODELS = [
        # Text Detection
        ModelInfo(
            name="Comic Text Detector",
            url="https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
            filename="comictextdetector.pt.onnx",
            category="text_detection",
            size_mb=11.5
        ),
        
        # OCR Models
        ModelInfo(
            name="Manga OCR Config",
            url="https://huggingface.co/kha-white/manga-ocr-base/resolve/main/config.json",
            filename="config.json",
            category="ocr",
            size_mb=0.001
        ),
        ModelInfo(
            name="Manga OCR Preprocessor Config",
            url="https://huggingface.co/kha-white/manga-ocr-base/resolve/main/preprocessor_config.json",
            filename="preprocessor_config.json",
            category="ocr",
            size_mb=0.001
        ),
        ModelInfo(
            name="Manga OCR Model",
            url="https://huggingface.co/kha-white/manga-ocr-base/resolve/main/pytorch_model.bin",
            filename="pytorch_model.bin",
            category="ocr",
            size_mb=450.0
        ),
        ModelInfo(
            name="Manga OCR Special Tokens",
            url="https://huggingface.co/kha-white/manga-ocr-base/resolve/main/special_tokens_map.json",
            filename="special_tokens_map.json",
            category="ocr",
            size_mb=0.001
        ),
        ModelInfo(
            name="Manga OCR Tokenizer Config",
            url="https://huggingface.co/kha-white/manga-ocr-base/resolve/main/tokenizer_config.json",
            filename="tokenizer_config.json",
            category="ocr",
            size_mb=0.001
        ),
        
        # Segmentation Models
        ModelInfo(
            name="YOLOv8 Bubble Segmentation Config",
            url="https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/config.yaml",
            filename="config.yaml",
            category="segmentation",
            size_mb=0.001
        ),
        ModelInfo(
            name="YOLOv8 Bubble Segmentation Model",
            url="https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble/resolve/main/model_dynamic.onnx",
            filename="model_dynamic.onnx",
            category="segmentation",
            size_mb=52.0
        ),
    ]
    
    CONFIG_FILE = "model_config.json"
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.model_path = None
        self.load_config()
        
    def load_config(self):
        """Load model path configuration"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self.model_path = data.get('model_path')
        except Exception as e:
            self.logger.error(f"Error loading model config: {e}")
    
    def save_config(self):
        """Save model path configuration"""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump({'model_path': self.model_path}, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving model config: {e}")
    
    def set_model_path(self, path: str):
        """Set và lưu model path"""
        self.model_path = path
        self.save_config()
        
    def get_model_path(self) -> Optional[str]:
        """Lấy model path hiện tại"""
        return self.model_path
    
    def check_missing_models(self) -> Dict[str, List[ModelInfo]]:
        """Kiểm tra các model còn thiếu theo category"""
        if not self.model_path or not os.path.exists(self.model_path):
            # Trả về tất cả models nếu chưa có path
            return self._group_models_by_category(self.REQUIRED_MODELS)
        
        missing_models = []
        for model in self.REQUIRED_MODELS:
            file_path = os.path.join(self.model_path, model.filename)
            if not os.path.exists(file_path):
                missing_models.append(model)
        
        return self._group_models_by_category(missing_models)
    
    def _group_models_by_category(self, models: List[ModelInfo]) -> Dict[str, List[ModelInfo]]:
        """Nhóm models theo category"""
        grouped = {}
        for model in models:
            if model.category not in grouped:
                grouped[model.category] = []
            grouped[model.category].append(model)
        return grouped
    
    def get_total_download_size(self, models: List[ModelInfo]) -> float:
        """Tính tổng dung lượng cần tải (MB)"""
        return sum(m.size_mb or 0 for m in models)
    
    def is_setup_complete(self) -> bool:
        """Kiểm tra xem đã setup đầy đủ models chưa"""
        missing = self.check_missing_models()
        return len(missing) == 0 and self.model_path is not None