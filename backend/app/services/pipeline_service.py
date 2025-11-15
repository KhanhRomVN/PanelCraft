import os
import cv2
import numpy as np
import uuid
from typing import List, Dict, Any, Optional
import logging

from app.schemas.pipeline import ProcessingStep, ImageResult, SegmentData, OCRResult
from app.services.segmentation_service import SegmentationService
from app.services.text_detection_service import TextDetectionService
from app.services.ocr_service import OCRService
from app.services.inpainting_service import InpaintingService
from app.utils.image_utils import numpy_to_base64, save_temp_image
from app.core.config import settings

logger = logging.getLogger(__name__)

class MangaPipelineService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.segmentation_service = SegmentationService(model_base_path)
        self.text_detection_service = TextDetectionService(model_base_path)
        self.ocr_service = OCRService(model_base_path)
        self.inpainting_service = InpaintingService(model_base_path)
        
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
            cleaned_image = None  # Khởi tạo biến cleaned_image
            results = ImageResult(
                image_index=0,
                original_path=image_path,
                original_dimensions=(image_rgb.shape[1], image_rgb.shape[0]),
                segments=[],
                ocr_results=[]
            )
            
            # Run pipeline steps
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.SEGMENTATION in steps:
                logger.info(f"[STEP1] ═══════════════════════════════════════════════════════")
                logger.info(f"[STEP1] Starting bubble segmentation...")
                
                # Step 1: Segmentation (không tính rectangle ngay)
                segments_data, _, masks = await self.segmentation_service.process(image_rgb)
                results.segments = segments_data
                
                logger.info(f"[STEP1] Detected {len(segments_data)} bubble segments")
                logger.info(f"[STEP1] Completed ✓")
            
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.TEXT_DETECTION in steps:
                logger.info(f"[STEP2] ═══════════════════════════════════════════════════════")
                logger.info(f"[STEP2] Starting text detection in segments...")
                
                # Step 2A: Detect text boxes TRONG từng segment (để tính rectangle)
                segments_with_mask = []
                
                for i, seg in enumerate(segments_data):
                    segments_with_mask.append({
                        'id': seg.id,
                        'box': seg.box,
                        'mask': masks[i]
                    })
                
                text_boxes_per_segment = await self.text_detection_service.detect_text_in_segments(
                    image_rgb,
                    segments_with_mask
                )
                
                logger.info(f"[STEP2] Text detection in segments completed")
                
                # Step 2B: Tính rectangles dựa vào text boxes
                logger.info(f"[STEP2] Calculating rectangles based on text boxes...")
                
                segments_data = self.segmentation_service.calculate_rectangles_with_text_boxes(
                    segments_data,
                    masks,
                    text_boxes_per_segment,
                    image_rgb
                )
                
                results.segments = segments_data
                
                # STEP 1 VISUALIZATION (chỉ boundaries, không cần rectangles)
                vis_boundaries = self.segmentation_service._create_step1_visualization(
                    image_rgb.copy(), segments_data, masks
                )
                
                # Save Step 1 Vis: Boundaries only
                step1_vis_path = save_temp_image(vis_boundaries, "step1_boundaries")
                step1_vis_url = f"/temp/{os.path.basename(step1_vis_path)}"
                
                logger.info(f"[STEP1] Visualization (Green boundaries): {step1_vis_url}")
                
                # Extract rectangles metadata (hỗ trợ nhiều rectangles cho 1 segment)
                rectangles = []
                for seg in segments_data:
                    for rect_idx, rectangle in enumerate(seg.rectangles):
                        x, y, w, h = rectangle
                        rectangles.append({
                            'segment_id': seg.id,
                            'rect_id': rect_idx,
                            'x': x,
                            'y': y,
                            'w': w, 
                            'h': h
                        })
                results.rectangles = rectangles
                results.original_dimensions = (image_rgb.shape[1], image_rgb.shape[0])
                
                # Step 2C: Detect ALL text boxes trên ảnh gốc
                all_text_boxes, all_text_scores = await self.text_detection_service.detect_all_text_boxes(image_rgb)
                logger.info(f"[STEP2] Detected {len(all_text_boxes)} text boxes globally")
                
                # Step 2D: Clean text TRONG bubble segments  
                cleaned_image, blank_canvas_vis, text_vis = await self.text_detection_service.process_segments(
                    image_rgb, 
                    segments_with_mask,
                    text_boxes_per_segment
                )
                
                logger.info(f"[STEP2] DEBUG: text_boxes_per_segment passed to process_segments: {len(text_boxes_per_segment)} segments")
                for idx, boxes in enumerate(text_boxes_per_segment):
                    logger.info(f"[STEP2] DEBUG:   Segment #{idx}: {len(boxes)} boxes")
                
                # STEP 2 VISUALIZATION
                # Save Step 2 Vis 1: Blank canvas với green boundaries
                step2_vis1_path = save_temp_image(blank_canvas_vis, "step2_blank_canvas")
                step2_vis1_url = f"/temp/{os.path.basename(step2_vis1_path)}"
                
                # Save Step 2 Vis 2: Text masks + boxes
                step2_vis2_path = save_temp_image(text_vis, "step2_text_masks")
                step2_vis2_url = f"/temp/{os.path.basename(step2_vis2_path)}"
                
                # Lưu cleaned image
                if cleaned_image is not None:
                    td_path = save_temp_image(cleaned_image, "cleaned_text")
                    results.cleaned_text_result = f"/temp/{os.path.basename(td_path)}"
                
                logger.info(f"[STEP2] Visualization 1 (Blank canvas with boundaries): {step2_vis1_url}")
                logger.info(f"[STEP2] Visualization 2 (Text masks + boxes): {step2_vis2_url}")
                logger.info(f"[STEP2] Cleaned image: {results.cleaned_text_result}")
                
                # STEP 2 VISUALIZATION 3: Rectangles (màu đỏ) - TẠO SAU VIS 1 & 2
                try:
                    vis_rectangles = self.segmentation_service._create_step2_visualization3(
                        image_rgb.copy(), segments_data, masks
                    )
                    
                    # Save Step 2 Vis 3: Rectangles overlay
                    step2_vis3_path = save_temp_image(vis_rectangles, "step2_rectangles")
                    step2_vis3_url = f"/temp/{os.path.basename(step2_vis3_path)}"
                    
                    logger.info(f"[STEP2] Visualization 3 (Rectangles overlay): {step2_vis3_url}")
                except Exception as e:
                    logger.error(f"[STEP2] Error creating Visualization 3: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                logger.info(f"[STEP2] Completed ✓")
                
                # Step 2B: Filter để tìm text boxes NGOÀI bubble segments
                text_boxes_outside = self.text_detection_service.filter_boxes_outside_segments(
                    all_text_boxes, 
                    all_text_scores,
                    segments_data
                )
                
                logger.info(f"[STEP2] Found {len(text_boxes_outside)} text boxes outside bubbles")
                
                # Step 2B-NEW: Xử lý text OUTSIDE bubbles
                if len(text_boxes_outside) > 0 and cleaned_image is not None:
                    # Filter scores tương ứng
                    text_scores_outside = all_text_scores[[i for i, box in enumerate(all_text_boxes) if any(np.array_equal(box, ob) for ob in text_boxes_outside)]]
                    
                    text_outside_data, vis1_masks, vis2_black_canvas, vis3_filtered, vis4_filtered_masks, vis5_inpainted = await self.text_detection_service.process_text_outside_bubbles(
                        cleaned_image,
                        text_boxes_outside,
                        text_scores_outside,
                        self.ocr_service,
                        self.inpainting_service
                    )
                    
                    # Lưu 5 visualizations
                    if vis1_masks is not None:
                        vis1_path = save_temp_image(vis1_masks, "text_outside_vis1_masks")
                        vis1_url = f"/temp/{os.path.basename(vis1_path)}"
                        logger.info(f"[STEP2] Text Outside VIS 1 (Masks on cleaned image): {vis1_url}")
                    
                    if vis2_black_canvas is not None:
                        vis2_path = save_temp_image(vis2_black_canvas, "text_outside_vis2_black_canvas")
                        vis2_url = f"/temp/{os.path.basename(vis2_path)}"
                        logger.info(f"[STEP2] Text Outside VIS 2 (Black canvas with text masks): {vis2_url}")
                    
                    if vis3_filtered is not None:
                        vis3_path = save_temp_image(vis3_filtered, "text_outside_vis3_filtered")
                        vis3_url = f"/temp/{os.path.basename(vis3_path)}"
                        logger.info(f"[STEP2] Text Outside VIS 3 (Filtered boxes with OCR): {vis3_url}")
                    
                    if vis4_filtered_masks is not None:
                        vis4_path = save_temp_image(vis4_filtered_masks, "text_outside_vis4_filtered_masks")
                        vis4_url = f"/temp/{os.path.basename(vis4_path)}"
                        logger.info(f"[STEP2] Text Outside VIS 4 (Filtered masks on cleaned image): {vis4_url}")
                    
                    if vis5_inpainted is not None:
                        vis5_path = save_temp_image(vis5_inpainted, "text_outside_vis5_inpainted")
                        vis5_url = f"/temp/{os.path.basename(vis5_path)}"
                        logger.info(f"[STEP2] Text Outside VIS 5 (Inpainted result): {vis5_url}")                
                
            if ProcessingStep.FULL_PIPELINE in steps or ProcessingStep.OCR in steps:
                logger.info(f"[STEP3] ═══════════════════════════════════════════════════════")
                logger.info(f"[STEP3] Starting OCR for bubble segments...")
                
                # Step 3: OCR
                if results.segments:
                    ocr_results = await self.ocr_service.process_segments(image_rgb, results.segments)
                    results.ocr_results = ocr_results
                    
                    logger.info(f"[STEP3] Completed OCR for {len(ocr_results)} segments")
                    logger.info(f"[STEP3] Completed ✓")
            
            # Set image_index properly
            results.image_index = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
