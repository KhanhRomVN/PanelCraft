import os
import cv2
import numpy as np
import uuid
from typing import List, Dict, Any, Optional
import logging

from app.models.pipeline_models import ProcessingStep, ImageResult, SegmentData, OCRResult
from app.services.segmentation_service import SegmentationService
from app.services.text_detection_service import TextDetectionService
from app.services.ocr_service import OCRService
from app.utils.image_utils import numpy_to_base64, save_temp_image
from app.core.config import settings

logger = logging.getLogger(__name__)

class MangaPipelineService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.segmentation_service = SegmentationService(model_base_path)
        self.text_detection_service = TextDetectionService(model_base_path)
        self.ocr_service = OCRService(model_base_path)
        
    async def process_single_image(
        self, 
        image_path: str, 
        steps: List[ProcessingStep] = None,
        options: Dict[str, Any] = None
    ) -> ImageResult:
        """
        Process a single image through the specified steps
        """
        if steps is None:
            steps = [ProcessingStep.FULL_PIPELINE]
            
        if options is None:
            options = {}
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = ImageResult(
                image_index=0,
                original_path=image_path,
                original_dimensions=(image_rgb.shape[1], image_rgb.shape[0]),
                segments=[],
                ocr_results=[]
            )
            
            # Run pipeline steps
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.SEGMENTATION in steps:
                # Step 1: Segmentation
                segments_data, visualization_img, masks = await self.segmentation_service.process(image_rgb)
                results.segments = segments_data
                
                print(f"[SEGMENTATION] Detected {len(segments_data)} segments after NMS (confidence >= 0.5)")
                
                # THÊM: Lưu visualization image
                if visualization_img is not None:
                    vis_path = save_temp_image(visualization_img, "visualization")
                    vis_url = f"/temp/{os.path.basename(vis_path)}"
                    print(f"[SEGMENTATION] Saved visualization with segments outline to: {vis_url}")
                
                # THÊM: Extract rectangles metadata
                rectangles = []
                for seg in segments_data:
                    if seg.rectangle:
                        x, y, w, h = seg.rectangle
                        rectangles.append({
                            'id': seg.id,
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h
                        })
                results.rectangles = rectangles
                
                # Lưu original dimensions
                results.original_dimensions = (image_rgb.shape[1], image_rgb.shape[0])
            
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.TEXT_DETECTION in steps:
                # Step 2: Text Detection - xử lý segments để remove text
                # Sử dụng masks từ segmentation
                segments_with_mask = []
                
                for i, seg in enumerate(segments_data):
                    segments_with_mask.append({
                        'id': seg.id,
                        'box': seg.box,
                        'mask': masks[i]
                    })
                
                cleaned_image = await self.text_detection_service.process_segments(
                    image_rgb, 
                    segments_with_mask
                )
                
                print(f"[Pipeline] Cleaned image shape: {cleaned_image.shape if cleaned_image is not None else 'None'}")
                
                if cleaned_image is not None:
                    td_path = save_temp_image(cleaned_image, "cleaned_text")
                    results.cleaned_text_result = f"/temp/{os.path.basename(td_path)}"
                    print(f"[Pipeline] Saved cleaned image to: {results.cleaned_text_result}")    
            
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.OCR in steps:
                # Step 3: OCR
                if results.segments:
                    ocr_results = await self.ocr_service.process_segments(image_rgb, results.segments)
                    results.ocr_results = ocr_results
            
            # Set image_index properly
            results.image_index = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise