"""
Application-wide custom exception hierarchy.

Purpose:
    Provide structured, typed errors across domains (image processing, batch ops, API layer)
    and unify HTTP/JSON response formatting in a single place.

Design:
    - BaseAppError: Root custom exception (never raised directly).
    - ValidationError: Input validation or precondition failures.
    - DomainError: Domain/service layer operational failures (model load, processing).
    - ResourceError: Missing filesystem / model resource issues.
    - PipelineStepError: Failures inside pipeline step execution.

Features:
    - Each exception carries: message, code (short slug), details (optional dict payload).
    - to_dict() / to_response() helpers for consistent JSON output.
    - __str__ overridden for readable logging.
    - HTTP status mapping defined in HTTP_STATUS_MAP (can be extended).

Usage Example:
    raise ValidationError("Invalid image size", details={"max_size": 2048, "got": 8192})

    In a FastAPI exception handler:
        @app.exception_handler(BaseAppError)
        async def app_error_handler(request: Request, exc: BaseAppError):
            return exc.to_response()

Extensibility:
    - Add more subclasses as required (e.g. AuthenticationError, ConcurrencyError).
    - Replace direct raises of ValueError / RuntimeError with appropriate custom classes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


class BaseAppError(Exception):
    """Root custom application exception."""

    default_code = "app_error"

    def __init__(self, message: str, *, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }

    def to_response(self, status_code: int | None = None) -> JSONResponse:
        status = status_code or HTTP_STATUS_MAP.get(self.code, 400)
        return JSONResponse(status_code=status, content=self.to_dict())

    def __str__(self) -> str:  # pragma: no cover - formatting
        if self.details:
            return f"{self.code}: {self.message} ({self.details})"
        return f"{self.code}: {self.message}"


class ValidationError(BaseAppError):
    """Raised when input validation or preconditions fail."""
    default_code = "validation_error"


class DomainError(BaseAppError):
    """Raised for domain/service layer processing failures."""
    default_code = "domain_error"


class ResourceError(BaseAppError):
    """Raised when required model/resource files are missing/unavailable."""
    default_code = "resource_error"


class PipelineStepError(BaseAppError):
    """Raised for failures inside pipeline step execution."""
    default_code = "pipeline_step_error"


# HTTP status code mapping per error code (extend as needed)
HTTP_STATUS_MAP: Dict[str, int] = {
    "app_error": 400,
    "validation_error": 422,
    "domain_error": 500,
    "resource_error": 404,
    "pipeline_step_error": 500,
}


__all__ = [
    "BaseAppError",
    "ValidationError",
    "DomainError",
    "ResourceError",
    "PipelineStepError",
    "HTTP_STATUS_MAP",
]
