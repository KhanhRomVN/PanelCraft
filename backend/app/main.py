from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
from typing import List, Optional
import logging

from app.api.routers.processing import router as processing_router
from app.core.config import settings
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Manga Processing Backend",
    description="Backend API for manga image processing pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:5174",  # Alternative Vite port
        "http://localhost:5175",  # Electron dev
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include API routes
# Register routers (extendable: add more domain-specific routers here)
app.include_router(processing_router, prefix="/api/v1")

# Create temp directory
temp_dir = os.path.abspath("temp")
os.makedirs(temp_dir, exist_ok=True)

# Verify temp directory exists and is writable
if not os.path.exists(temp_dir):
    raise RuntimeError(f"Failed to create temp directory: {temp_dir}")
if not os.access(temp_dir, os.W_OK):
    raise RuntimeError(f"Temp directory is not writable: {temp_dir}")

print(f"[Server] Temp directory: {temp_dir}")
app.mount("/temp", StaticFiles(directory=temp_dir), name="temp")

@app.get("/")
async def root():
    return {"message": "Manga Processing Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test")
async def test_route():
    return {"message": "Hello world!"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
