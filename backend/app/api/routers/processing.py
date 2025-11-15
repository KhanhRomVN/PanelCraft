# app/api/routers/processing.py
"""
Processing & image-related API routes.

Refactored from the previous monolithic app/api/endpoints.py:
- Clear separation of concerns
- Uses unified schemas from app.schemas.pipeline
- Background batch processing state moved to a simple in-memory store (app/state/batch_store.py)

NOTE:
In production replace in-memory batch store with Redis or persistent backend.
"""

from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List

import os
import shutil
import uuid
import logging

from app.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    BatchResponse,
    ProcessingStep,
    ImageResult,
)
from app.services.pipeline_service import MangaPipelineService
from app.core.config import settings
from app.state.batch_store import batch_processing_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["processing"])


@router.post("/process-images", response_model=PipelineResponse)
async def process_images(request: PipelineRequest):
    """
    Process a batch of images through the pipeline synchronously.

    Returns PipelineResponse containing ImageResult list under data['results'].
    """
    try:
        if not os.path.exists(request.model_base_path):
            return PipelineResponse(
                success=False,
                message="Model path not found",
                error=f"Model directory not found: {request.model_base_path}",
            )

        pipeline_service = MangaPipelineService(request.model_base_path)

        results: List[ImageResult] = []
        for i, image_path in enumerate(request.image_paths):
            if not os.path.exists(image_path):
                logger.warning(f"[process-images] Skipping missing image: {image_path}")
                continue

            try:
                result = await pipeline_service.process_single_image(
                    image_path,
                    steps=request.steps,
                    options=request.options or {},
                )
                result.image_index = i
                results.append(result)
            except Exception as e:
                logger.error(f"[process-images] Error processing {image_path}: {e}")
                continue

        return PipelineResponse(
            success=True,
            message=f"Processed {len(results)} images successfully",
            data={"results": results},
        )

    except Exception as e:
        logger.exception("[process-images] Pipeline failed")
        return PipelineResponse(
            success=False,
            message="Processing failed",
            error=str(e),
        )


@router.post("/upload-and-process")
async def upload_and_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_base_path: str = "models",
    steps: List[ProcessingStep] = [ProcessingStep.FULL_PIPELINE],
):
    """
    Upload multiple images, then process them asynchronously (background task).
    Returns a request_id that can be polled via /batch-status/{request_id}.
    """
    request_id = str(uuid.uuid4())

    temp_paths: List[str] = []
    for file in files:
        temp_path = os.path.join(settings.TEMP_DIR, f"{request_id}_{file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_paths.append(temp_path)

    # Initialize store entry
    batch_processing_store[request_id] = {
        "status": "processing",
        "total_images": len(temp_paths),
        "processed_images": 0,
        "results": [],
    }

    background_tasks.add_task(
        process_images_background,
        request_id,
        temp_paths,
        model_base_path,
        steps,
    )

    return {
        "request_id": request_id,
        "status": "processing",
        "message": f"Started processing {len(temp_paths)} images",
    }


@router.get("/batch-status/{request_id}", response_model=BatchResponse)
async def get_batch_status(request_id: str):
    """
    Poll the status of a background batch processing request.
    """
    if request_id not in batch_processing_store:
        raise HTTPException(status_code=404, detail="Request ID not found")

    batch_data = batch_processing_store[request_id]
    return BatchResponse(
        request_id=request_id,
        status=batch_data["status"],
        results=batch_data["results"],
        total_images=batch_data["total_images"],
        processed_images=batch_data["processed_images"],
    )


@router.post("/single-image")
async def process_single_image(
    file: UploadFile = File(...),
    model_base_path: str = "models",
    steps: List[ProcessingStep] = [ProcessingStep.FULL_PIPELINE],
):
    """
    Upload and process a single image synchronously.
    """
    try:
        temp_path = os.path.join(
            settings.TEMP_DIR, f"single_{uuid.uuid4()}_{file.filename}"
        )
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline_service = MangaPipelineService(model_base_path)
        result = await pipeline_service.process_single_image(temp_path, steps=steps)

        os.remove(temp_path)
        return result
    except Exception as e:
        logger.exception("[single-image] Failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-models")
async def check_models(model_base_path: str):
    """
    Validate required model files exist.
    """
    try:
        segmentation_model = os.path.join(
            model_base_path, "segmentation", settings.SEGMENTATION_MODEL.split("/")[-1]
        )
        text_detection_model = os.path.join(
            model_base_path,
            "text_detection",
            settings.TEXT_DETECTION_MODEL.split("/")[-1],
        )

        return {
            "segmentation_exists": os.path.exists(segmentation_model),
            "text_detection_exists": os.path.exists(text_detection_model),
            "model_base_path": model_base_path,
        }
    except Exception as e:
        logger.exception("[check-models] Failed")
        raise HTTPException(status_code=500, detail=str(e))


async def process_images_background(
    request_id: str,
    image_paths: List[str],
    model_base_path: str,
    steps: List[ProcessingStep],
):
    """
    Background task to process uploaded images.
    Updates the in-memory batch store incrementally.
    """
    try:
        pipeline_service = MangaPipelineService(model_base_path)
        results: List[ImageResult] = []

        for i, image_path in enumerate(image_paths):
            try:
                result = await pipeline_service.process_single_image(
                    image_path, steps=steps
                )
                result.image_index = i
                results.append(result)

                batch_processing_store[request_id]["processed_images"] = i + 1
                batch_processing_store[request_id]["results"] = results
            except Exception as e:
                logger.error(f"[background] Error processing {image_path}: {e}")
                continue

        batch_processing_store[request_id]["status"] = "completed"

    except Exception as e:
        logger.exception("[background] Batch processing failed")
        batch_processing_store[request_id]["status"] = "error"
        batch_processing_store[request_id]["error"] = str(e)
