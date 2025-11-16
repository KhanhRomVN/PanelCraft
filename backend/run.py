#!/usr/bin/env python3
"""
Run script for the manga processing backend
"""
import uvicorn
from app.main import app
from app.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )