# app/api/endpoints.py
"""
DEPRECATED MODULE

This file previously contained all processing-related API routes in a single
monolithic router. The implementation has been refactored and split into
modular routers located under:

    app/api/routers/processing.py

All new code SHOULD import from the routers package, for example:

    from app.api.routers.processing import router as processing_router

If legacy imports still reference `app.api.endpoints`, they will continue to
work because we re-export the new processing router below. Please migrate
those imports to the new path and then remove this file entirely once no
longer needed.

To find remaining usages you can run a regex search for:
    app\.api\.endpoints

After all usages are updated, delete this file.
"""

from __future__ import annotations

from fastapi import APIRouter
from app.api.routers.processing import router as processing_router

# Re-export the processing routes for backward compatibility.
router = APIRouter()
router.include_router(processing_router)

__all__ = ["router"]
