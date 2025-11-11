from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
from typing import List

from app.models.pipeline_models import (
    PipelineRequest, PipelineResponse, BatchResponse, 
    ProcessingStep, ImageResult
)
from app.services.pipeline_service import MangaPipelineService
from app.core.config import settings

router = APIRouter()

# In-memory storage for batch processing (use Redis in production)
batch_processing = {}

@router.post("/process-images", response_model=PipelineResponse)
async def process_images(request: PipelineRequest):
    """
    Process a batch of images through the manga pipeline
    """
    try:
        print(f"[API] Received request to process {len(request.image_paths)} images")
        print(f"[API] Model base path: {request.model_base_path}")
        print(f"[API] Steps: {request.steps}")
        
        # Validate model paths
        if not os.path.exists(request.model_base_path):
            return PipelineResponse(
                success=False,
                message="Model path not found",
                error=f"Model directory not found: {request.model_base_path}"
            )
        
        # Initialize pipeline service
        pipeline_service = MangaPipelineService(request.model_base_path)
        
        # Process images
        results = []
        for i, image_path in enumerate(request.image_paths):
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
                
            try:
                result = await pipeline_service.process_single_image(
                    image_path, 
                    steps=request.steps,
                    options=request.options or {}
                )
                # Set correct image index
                result.image_index = i
                results.append(result)
            except Exception as e:
                # Continue with other images if one fails
                print(f"Error processing image {image_path}: {e}")
                continue
        
        return PipelineResponse(
            success=True,
            message=f"Processed {len(results)} images successfully",
            data={"results": results}
        )
        
    except Exception as e:
        return PipelineResponse(
            success=False,
            message="Processing failed",
            error=str(e)
        )

@router.post("/upload-and-process")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_base_path: str = "models",
    steps: List[ProcessingStep] = [ProcessingStep.FULL_PIPELINE]
):
    """
    Upload images and process them
    """
    request_id = str(uuid.uuid4())
    
    # Save uploaded files temporarily
    temp_paths = []
    for file in files:
        temp_path = os.path.join(settings.TEMP_DIR, f"{request_id}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_paths.append(temp_path)
    
    # Initialize batch processing
    batch_processing[request_id] = {
        "status": "processing",
        "total_images": len(temp_paths),
        "processed_images": 0,
        "results": []
    }
    
    # Process in background
    background_tasks.add_task(
        process_images_background,
        request_id,
        temp_paths,
        model_base_path,
        steps
    )
    
    return {
        "request_id": request_id,
        "status": "processing",
        "message": f"Started processing {len(temp_paths)} images"
    }

@router.get("/batch-status/{request_id}", response_model=BatchResponse)
async def get_batch_status(request_id: str):
    """
    Get status of a batch processing job
    """
    if request_id not in batch_processing:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    batch_data = batch_processing[request_id]
    
    return BatchResponse(
        request_id=request_id,
        status=batch_data["status"],
        results=batch_data["results"],
        total_images=batch_data["total_images"],
        processed_images=batch_data["processed_images"]
    )

@router.post("/single-image")
async def process_single_image(
    file: UploadFile = File(...),
    model_base_path: str = "models",
    steps: List[ProcessingStep] = [ProcessingStep.FULL_PIPELINE]
):
    """
    Process a single image
    """
    try:
        # Save uploaded file
        temp_path = os.path.join(settings.TEMP_DIR, f"single_{uuid.uuid4()}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image
        pipeline_service = MangaPipelineService(model_base_path)
        result = await pipeline_service.process_single_image(
            temp_path, 
            steps=steps
        )
        
        # Clean up
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_images_background(
    request_id: str,
    image_paths: List[str],
    model_base_path: str,
    steps: List[ProcessingStep]
):
    """
    Background task for processing images
    """
    try:
        pipeline_service = MangaPipelineService(model_base_path)
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                result = await pipeline_service.process_single_image(
                    image_path, 
                    steps=steps
                )
                results.append(result)
                
                # Update progress
                batch_processing[request_id]["processed_images"] = i + 1
                batch_processing[request_id]["results"] = results
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        # Mark as completed
        batch_processing[request_id]["status"] = "completed"
        
    except Exception as e:
        batch_processing[request_id]["status"] = "error"
        batch_processing[request_id]["error"] = str(e)
        
@router.get("/check-models")
async def check_models(model_base_path: str):
    """
    Check if models exist and are valid
    """
    try:
        segmentation_model = os.path.join(model_base_path, "segmentation", "manga_bubble_seg.onnx")
        text_detection_model = os.path.join(model_base_path, "text_detection", "comictextdetector.pt.onnx")
        
        return {
            "segmentation_exists": os.path.exists(segmentation_model),
            "text_detection_exists": os.path.exists(text_detection_model),
            "model_base_path": model_base_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))