import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = True
    
    # Model Paths
    MODEL_BASE_PATH: str = "models"
    SEGMENTATION_MODEL: str = "segmentation/manga_bubble_seg.onnx"
    TEXT_DETECTION_MODEL: str = "text_detection/comictextdetector.pt.onnx"
    
    # Processing Settings
    MAX_IMAGE_SIZE: int = 2048
    TEMP_DIR: str = "temp"
    
    # OCR Settings
    OCR_MAX_TEXT_LENGTH: int = 1000
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Bỏ qua các field không khai báo trong .env

settings = Settings()