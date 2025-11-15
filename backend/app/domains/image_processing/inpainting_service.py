"""
Inpainting Service.

Responsibilities:
- Load and patch lama-cleaner to force usage of a local LaMa (text_inpainting_manga.pt) model file
  (prevent any network downloads).
- Provide high-level async API `inpaint_text_regions` to remove text given a binary mask.
- Handle device (CPU/CUDA) selection and verbose diagnostic logging.
- Enforce tunable thresholds via centralized constants (see app.config.constants).

Refactor / Clean Code Improvements:
- Added comprehensive module, class & method docstrings.
- Normalized logging prefixes: [Inpainting Init], [Inpainting Load], [Inpainting].
- Removed duplicated directory typo handling (original had same path twice).
- Extracted constant values from hard-coded numbers (INPAINT_DEFAULT_DILATE_KERNEL, INPAINT_MIN_MASK_COVERAGE_PERCENT).
- Added early returns and clearer error messaging.
- Reduced overly noisy logging while retaining key diagnostics (can tune further).
- Ensured graceful degradation: if model cannot load, method returns original image.
- Added type hints throughout.
- Simplified patching logic & clarified critical patch steps.

Future Enhancements (not implemented now to preserve behavior):
- Introduce an InpaintingModel abstraction (see app/core/ml_models/base.py).
- Add caching mechanism when multiple masks processed on same image.
- Support alternative inpainting backends (diffusers-based / SD based).
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import logging
from typing import Optional
from app.config import constants as C

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class InpaintingService:
    """
    High-level wrapper around LaMa (via lama-cleaner) for text region inpainting.

    Args:
        model_base_path: Root path containing subfolder 'inpainting' with model file
                         'text_inpainting_manga.pt'.

    Lifecycle:
        Instantiation triggers immediate model loading + patching of lama-cleaner internals
        to avoid network calls.
    """

    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.model = None
        self.model_manager = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("[Inpainting Init] Device: %s (CUDA=%s)", self.device, torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                logger.info("[Inpainting Init] CUDA device name: %s", torch.cuda.get_device_name(0))
            except Exception:
                logger.info("[Inpainting Init] Unable to get CUDA device name")

        self._load_model()

    # -------------------------------------------------------------------------
    # MODEL LOADING / PATCHING
    # -------------------------------------------------------------------------
    def _load_model(self) -> None:
        """
        Load LaMa model using lama-cleaner with custom patching to enforce local file usage.

        Patch strategy:
            - Override helper.load_jit_model & helper.download_model to return local path.
            - Override model.lama.load_jit_model similarly if accessible.
            - Force LAMA_MODEL_URL to point to file:// path (if attribute exists).
        """
        logger.info("[Inpainting Load] Starting model loading...")
        model_dir = os.path.join(self.model_base_path, "inpainting")
        model_path = os.path.join(model_dir, "text_inpainting_manga.pt")

        if not os.path.exists(model_dir):
            logger.error("[Inpainting Load] Model directory missing: %s", model_dir)
            self.model = None
            self.model_manager = None
            return

        if not os.path.exists(model_path):
            logger.error("[Inpainting Load] Model file not found: %s", model_path)
            self.model = None
            self.model_manager = None
            return

        # Log directory contents for diagnostics
        try:
            files = os.listdir(model_dir)
            logger.info("[Inpainting Load] Directory contains %d file(s)", len(files))
            for f in files:
                fp = os.path.join(model_dir, f)
                size_mb = os.path.getsize(fp) / (1024 * 1024)
                logger.info("[Inpainting Load]   - %s (%.2f MB)", f, size_mb)
        except Exception as e:
            logger.warning("[Inpainting Load] Cannot list directory: %s", e)

        # Define patching helpers
        try:
            import lama_cleaner.helper as helper_module

            class _CustomLoader:
                def __init__(self, local_path: str):
                    self.local_path = local_path

                def __call__(self, _url_or_path: str, device, model_md5: str = None):
                    logger.info("[Inpainting Patch] Redirecting model load to local: %s", self.local_path)
                    return torch.jit.load(self.local_path, map_location=device)

            def _download_model_patch(url: str) -> str:
                logger.info("[Inpainting Patch] Intercepted download_model(%s) -> %s", url, model_path)
                return model_path

            helper_module.load_jit_model = _CustomLoader(model_path)  # type: ignore[attr-defined]
            helper_module.download_model = _download_model_patch  # type: ignore[attr-defined]
            logger.info("[Inpainting Load] Patched lama_cleaner.helper")
        except Exception as e:
            logger.warning("[Inpainting Load] Helper pre-patch failed: %s", e)

        try:
            import lama_cleaner.model.lama as lama_module

            if hasattr(lama_module, "load_jit_model"):
                lama_module.load_jit_model = _CustomLoader(model_path)  # type: ignore[attr-defined]
                logger.info("[Inpainting Load] Patched lama_cleaner.model.lama.load_jit_model")

            if hasattr(lama_module, "LAMA_MODEL_URL"):
                original_url = getattr(lama_module, "LAMA_MODEL_URL")
                lama_module.LAMA_MODEL_URL = f"file://{model_path}"  # type: ignore[attr-defined]
                logger.info("[Inpainting Load] Repointed LAMA_MODEL_URL: %s -> file://%s", original_url, model_path)
        except Exception as e:
            logger.warning("[Inpainting Load] lama module patch failed: %s", e)

        # Import config / manager
        try:
            from lama_cleaner.model_manager import ModelManager
            from lama_cleaner.schema import Config  # noqa: F401 (used later for inference)
            logger.info("[Inpainting Load] Imported ModelManager and Config")
        except Exception as e:
            logger.error("[Inpainting Load] Failed to import lama_cleaner components: %s", e)
            self.model = None
            self.model_manager = None
            return

        try:
            self.model_manager = ModelManager(name="lama", device=self.device)
            self.model = self.model_manager.model
            if self.model is None:
                logger.error("[Inpainting Load] ModelManager.model is None after initialization")
            else:
                logger.info("[Inpainting Load] Model loaded successfully (type=%s)", type(self.model))
        except Exception as e:
            logger.error("[Inpainting Load] ModelManager initialization failed: %s", e)
            self.model_manager = None
            self.model = None

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    async def inpaint_text_regions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dilate_kernel_size: int = C.INPAINT_DEFAULT_DILATE_KERNEL,
    ) -> Optional[np.ndarray]:
        """
        Inpaint text regions specified by a binary mask (255 = text region).

        Workflow:
            1. Validate model availability; fallback return original image if unloaded.
            2. Reject low coverage masks (< INPAINT_MIN_MASK_COVERAGE_PERCENT).
            3. Optionally dilate mask to better encompass text edges.
            4. Run LaMa model_manager(image, mask, config) with sensible config defaults.
            5. Return inpainted RGB image (uint8).

        Args:
            image: RGB image (H,W,3)
            mask: Binary mask (H,W), values {0,255} OR {0,1}
            dilate_kernel_size: Kernel size (pixels) for dilation before inpainting.

        Returns:
            Inpainted image or original if model not available / errors encountered.
        """
        if self.model is None or self.model_manager is None:
            logger.debug("[Inpainting] Model not loaded; returning original image")
            return image

        try:
            # Normalize mask to 0/255 uint8
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255
            else:
                unique_vals = np.unique(mask)
                if not (unique_vals == 0).all() and not (unique_vals == 255).all():
                    mask = (mask > 0).astype(np.uint8) * 255

            coverage_pct = np.sum(mask > 0) / mask.size * 100.0
            logger.debug("[Inpainting] Mask coverage: %.2f%%", coverage_pct)

            if coverage_pct < (C.INPAINT_MIN_MASK_COVERAGE_PERCENT * 100.0):
                logger.debug(
                    "[Inpainting] Coverage below threshold (%.2f%% < %.2f%%); skipping",
                    coverage_pct,
                    C.INPAINT_MIN_MASK_COVERAGE_PERCENT * 100.0,
                )
                return image

            # Dilation (optional)
            if dilate_kernel_size > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size)
                )
                mask = cv2.dilate(mask, kernel, iterations=1)
                dilated_cov = np.sum(mask > 0) / mask.size * 100.0
                logger.debug("[Inpainting] Dilated coverage: %.2f%%", dilated_cov)

            # Ensure image dtype
            if image.dtype != np.uint8:
                logger.debug("[Inpainting] Converting image dtype %s -> uint8", image.dtype)
                image_input = (image.clip(0, 255)).astype(np.uint8)
            else:
                image_input = image

            from lama_cleaner.schema import Config  # Lazy import for runtime adjustments

            # Config tuned for relatively small masked text regions
            config = Config(
                ldm_steps=25,
                ldm_sampler="plms",
                hd_strategy="Original",
                hd_strategy_crop_margin=196,
                hd_strategy_crop_trigger_size=1280,
                hd_strategy_resize_limit=2048,
                prompt="",
                negative_prompt="",
                use_croper=False,
                croper_x=0,
                croper_y=0,
                croper_height=512,
                croper_width=512,
                sd_scale=1.0,
                sd_mask_blur=5,
                sd_strength=0.75,
                sd_steps=50,
                sd_guidance_scale=7.5,
                sd_sampler="ddim",
                sd_seed=42,
                sd_match_histograms=False,
                cv2_flag="INPAINT_NS",
                cv2_radius=5,
                paint_by_example_steps=50,
                paint_by_example_guidance_scale=7.5,
                paint_by_example_mask_blur=5,
                paint_by_example_seed=42,
                paint_by_example_match_histograms=False,
                paint_by_example_example_image=None,
                p2p_steps=50,
                p2p_image_guidance_scale=1.5,
                p2p_guidance_scale=7.5,
                controlnet_conditioning_scale=0.4,
                controlnet_method="control_v11p_sd15_canny",
            )

            result = self.model_manager(image_input, mask, config)  # type: ignore[call-arg]
            if result is None:
                logger.warning("[Inpainting] Model returned None; returning original")
                return image

            if result.shape != image_input.shape:
                logger.debug(
                    "[Inpainting] Result shape mismatch %s vs %s; resizing",
                    result.shape,
                    image_input.shape,
                )
                result = cv2.resize(
                    result, (image_input.shape[1], image_input.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            # Diff stats
            diff_pixels = int(np.sum(result != image_input))
            if diff_pixels == 0:
                logger.debug("[Inpainting] No pixel changes detected")
            else:
                diff_pct = diff_pixels / result.size * 100.0
                logger.debug("[Inpainting] Changed pixels: %d (%.2f%%)", diff_pixels, diff_pct)

            return result

        except Exception as e:  # noqa: BLE001
            logger.error("[Inpainting] Error during inpainting: %s", e)
            return image


__all__ = ["InpaintingService"]
