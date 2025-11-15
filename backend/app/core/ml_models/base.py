"""
Base ML model abstractions.

Defines common interfaces and lightweight helper mixins for segmentation, text detection,
OCR and inpainting models. These abstractions aim to:

- Standardize model loading (single `_load()` implementation per concrete model)
- Provide a unified `is_ready` property & error handling
- Offer clear typing for `run_inference` signatures
- Keep dependencies minimal (pure Python / numpy typing only). Framework specifics
  (onnxruntime / torch / opencv) stay inside concrete implementations.

The pipeline services should depend on these interfaces instead of concrete classes
to simplify testing & future model swaps.

NOTE:
Concrete model classes live close to their domain service OR inside a future
`app/core/ml_models/impl/` package. This file remains framework-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union


class ModelLoadError(RuntimeError):
    """Raised when an ML model fails to load properly."""


class SupportsInference(Protocol):
    """
    Structural typing protocol for inference-capable objects.
    Any object implementing `run_inference` with the matching signature will be accepted.
    """
    def run_inference(self, *inputs: Any, **kwargs: Any) -> Any: ...


class BaseModel(ABC):
    """
    Base interface for all ML model wrappers.

    Lifecycle:
        1. Instantiate with required path/config
        2. Call `load()` OR rely on constructor of concrete class to automatically load
        3. Check `is_ready`
        4. Call `run_inference()`

    Concrete classes SHOULD:
        - Implement `_load_core()` returning underlying session/nn.Module/etc.
        - Store the loaded artifact in `self._core`
        - Avoid heavy logic inside `__init__` beyond argument assignment.

    Error Handling:
        - Wrap framework exceptions into `ModelLoadError` where appropriate.
    """

    def __init__(self) -> None:
        self._core: Any = None
        self._loaded: bool = False

    @property
    def is_ready(self) -> bool:
        """Whether the underlying model artifact has been successfully loaded."""
        return self._loaded and self._core is not None

    def load(self) -> None:
        """Public load entry point. Idempotent."""
        if self.is_ready:
            return
        try:
            self._core = self._load_core()
            self._loaded = True
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load model: {e}") from e

    @abstractmethod
    def _load_core(self) -> Any:
        """Framework-specific loading logic (ONNX session, Torch module, etc.)."""

    @abstractmethod
    def run_inference(self, *inputs: Any, **kwargs: Any) -> Any:
        """Execute a forward pass. Inputs & outputs are framework-defined."""


class SegmentationModel(BaseModel, ABC):
    """
    Specialization for segmentation models.

    Contract:
        - `run_inference(image_tensor)` returns raw outputs needed for post-processing
        - Implementers SHOULD keep preprocessing outside (service handles it)
    """
    @abstractmethod
    def run_inference(self, image_tensor: Any) -> Any:  # type: ignore[override]
        ...


class TextDetectionModel(BaseModel, ABC):
    """
    Specialization for text detection.

    Contract:
        - `run_inference(image_tensor)` returns tuple(structured_boxes, mask, raw_scores/aux)
    """
    @abstractmethod
    def run_inference(self, image_tensor: Any) -> Tuple[Any, Any, Any]:  # type: ignore[override]
        ...


class OCRModel(BaseModel, ABC):
    """
    Specialization for OCR models.

    Contract:
        - `run_inference(image_crop)` returns recognized text (string) OR structured result
    """
    @abstractmethod
    def run_inference(self, image_crop: Any) -> Union[str, Dict[str, Any]]:  # type: ignore[override]
        ...


class InpaintingModel(BaseModel, ABC):
    """
    Specialization for inpainting models.

    Contract:
        - `run_inference(image, mask, **cfg)` returns inpainted image array
    """
    @abstractmethod
    def run_inference(self, image: Any, mask: Any, **cfg: Any) -> Any:  # type: ignore[override]
        ...


__all__ = [
    "ModelLoadError",
    "SupportsInference",
    "BaseModel",
    "SegmentationModel",
    "TextDetectionModel",
    "OCRModel",
    "InpaintingModel",
]
