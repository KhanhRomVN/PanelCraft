# app/services/inpainting_service.py
import os
import cv2
import numpy as np
import torch
import logging
from typing import Optional, Tuple
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class InpaintingService:
    def __init__(self, model_base_path: str):
        self.model_base_path = model_base_path
        self.model = None
        self.model_manager = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"[Inpainting Init] ════════════════════════════════════════")
        logger.info(f"[Inpainting Init] Initializing InpaintingService")
        logger.info(f"[Inpainting Init] Model base path: {model_base_path}")
        logger.info(f"[Inpainting Init] Device: {self.device}")
        logger.info(f"[Inpainting Init] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"[Inpainting Init] CUDA device: {torch.cuda.get_device_name(0)}")
        
        self._load_model()
        
        logger.info(f"[Inpainting Init] ════════════════════════════════════════")
    
    def _load_model(self):
        """Load LaMa inpainting model"""
        logger.info(f"[Inpainting Load] ════════════════════════════════════════")
        logger.info(f"[Inpainting Load] Starting model loading process...")
        
        model_dir_correct = os.path.join(self.model_base_path, "inpainting")
        model_dir_typo = os.path.join(self.model_base_path, "inpainting")
        
        if os.path.exists(model_dir_correct):
            model_dir = model_dir_correct
            logger.info(f"[Inpainting Load] Using correct directory: 'inpainting'")
        elif os.path.exists(model_dir_typo):
            model_dir = model_dir_typo
            logger.warning(f"[Inpainting Load] Using typo directory: 'inpainting' (should rename to 'inpainting')")
        else:
            model_dir = model_dir_correct
            logger.error(f"[Inpainting Load] Neither 'inpainting' nor 'inpainting' directory exists")
        
        model_path = os.path.join(model_dir, "text_inpainting_manga.pt")
        
        logger.info(f"[Inpainting Load] Model directory: {model_dir}")
        logger.info(f"[Inpainting Load] Expected model path: {model_path}")
        
        if not os.path.exists(model_dir):
            logger.error(f"[Inpainting Load] ✗ Model directory does not exist: {model_dir}")
            logger.error(f"[Inpainting Load] Please create directory: mkdir -p {model_dir}")
            self.model = None
            return
        else:
            logger.info(f"[Inpainting Load] ✓ Model directory exists")
        
        try:
            files_in_dir = os.listdir(model_dir)
            logger.info(f"[Inpainting Load] Files in directory ({len(files_in_dir)}):")
            for f in files_in_dir:
                file_path = os.path.join(model_dir, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"[Inpainting Load]   - {f} ({file_size:.2f} MB)")
        except Exception as e:
            logger.error(f"[Inpainting Load] ✗ Cannot list directory: {e}")
        
        if not os.path.exists(model_path):
            logger.error(f"[Inpainting Load] ✗ Model file not found: {model_path}")
            logger.error(f"[Inpainting Load] Please place 'text_inpainting_manga.pt' (big-lama model) in: {model_dir}")
            self.model = None
            return
        else:
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"[Inpainting Load] ✓ Model file found: {model_path}")
            logger.info(f"[Inpainting Load] ✓ Model size: {model_size:.2f} MB")
        
        try:
            logger.info(f"[Inpainting Load] Step 1: Patching lama_cleaner BEFORE any imports...")
            
            import sys
            
            class CustomLamaLoader:
                """Custom loader to intercept model loading"""
                def __init__(self, custom_model_path):
                    self.custom_model_path = custom_model_path
                
                def __call__(self, url_or_path, device, model_md5: str = None):
                    logger.info(f"[Inpainting Patch] ═══════════════════════════════════")
                    logger.info(f"[Inpainting Patch] Model load intercepted!")
                    logger.info(f"[Inpainting Patch] Requested: {url_or_path}")
                    logger.info(f"[Inpainting Patch] Redirecting to: {self.custom_model_path}")
                    model = torch.jit.load(self.custom_model_path, map_location=device)
                    logger.info(f"[Inpainting Patch] ✓ Custom model loaded successfully")
                    logger.info(f"[Inpainting Patch] ═══════════════════════════════════")
                    return model
            
            # CRITICAL: Định nghĩa custom_download_model TRƯỚC
            def custom_download_model(url):
                logger.info(f"[Inpainting Patch] download_model() intercepted!")
                logger.info(f"[Inpainting Patch] Requested URL: {url}")
                logger.info(f"[Inpainting Patch] Returning local model path: {model_path}")
                return model_path
            
            # CRITICAL: Patch TRƯỚC KHI import bất kỳ module nào của lama_cleaner
            logger.info(f"[Inpainting Load] Pre-patching helper module...")
            try:
                import lama_cleaner.helper
                lama_cleaner.helper.load_jit_model = CustomLamaLoader(model_path)
                logger.info(f"[Inpainting Load] ✓ Pre-patched lama_cleaner.helper.load_jit_model")
                
                # CRITICAL: Patch download_model() để trả về local path trực tiếp
                lama_cleaner.helper.download_model = custom_download_model
                logger.info(f"[Inpainting Load] ✓ Pre-patched lama_cleaner.helper.download_model")
            except (ImportError, AttributeError) as e:
                logger.warning(f"[Inpainting Load] Could not pre-patch lama_cleaner.helper: {e}")
            
            logger.info(f"[Inpainting Load] Pre-patching lama model module...")
            try:
                import lama_cleaner.model.lama as lama_module
                if hasattr(lama_module, 'load_jit_model'):
                    lama_module.load_jit_model = CustomLamaLoader(model_path)
                    logger.info(f"[Inpainting Load] ✓ Pre-patched lama_cleaner.model.lama.load_jit_model")
                
                # CRITICAL: Patch LAMA_MODEL_URL để trả về local path
                if hasattr(lama_module, 'LAMA_MODEL_URL'):
                    original_url = lama_module.LAMA_MODEL_URL
                    lama_module.LAMA_MODEL_URL = f"file://{model_path}"
                    logger.info(f"[Inpainting Load] ✓ Patched LAMA_MODEL_URL: {original_url} -> file://{model_path}")
            except (ImportError, AttributeError) as e:
                logger.warning(f"[Inpainting Load] Could not pre-patch lama_module: {e}")
            
            try:
                from lama_cleaner.schema import Config
                logger.info(f"[Inpainting Load] ✓ Config imported")
            except ImportError as import_err:
                logger.error(f"[Inpainting Load] ✗ Failed to import Config: {import_err}")
                raise
            
            logger.info(f"[Inpainting Load] Step 2: Initializing ModelManager...")
            logger.info(f"[Inpainting Load]   - name: 'lama'")
            logger.info(f"[Inpainting Load]   - device: {self.device}")
            
            try:
                from lama_cleaner.model_manager import ModelManager
                logger.info(f"[Inpainting Load] ✓ ModelManager imported")
            except ImportError as import_err:
                logger.error(f"[Inpainting Load] ✗ Failed to import ModelManager: {import_err}")
                raise
            
            try:
                self.model_manager = ModelManager(
                    name='lama',
                    device=self.device
                )
                logger.info(f"[Inpainting Load] ✓ ModelManager object created")
            except Exception as init_err:
                logger.error(f"[Inpainting Load] ✗ ModelManager initialization failed: {init_err}")
                import traceback
                logger.error(f"[Inpainting Load] Traceback:\n{traceback.format_exc()}")
                raise
            
            logger.info(f"[Inpainting Load] Step 3: Accessing model from ModelManager...")
            
            try:
                self.model = self.model_manager.model
                logger.info(f"[Inpainting Load] ✓ Model attribute accessed")
            except Exception as model_err:
                logger.error(f"[Inpainting Load] ✗ Failed to access model: {model_err}")
                import traceback
                logger.error(f"[Inpainting Load] Traceback:\n{traceback.format_exc()}")
                raise
            
            if self.model is not None:
                logger.info(f"[Inpainting Load] ✓ Model loaded successfully")
                logger.info(f"[Inpainting Load] Model type: {type(self.model)}")
                logger.info(f"[Inpainting Load] Model device: {self.device}")
                logger.info(f"[Inpainting Load] Model class: {self.model.__class__.__name__}")
            else:
                logger.error(f"[Inpainting Load] ✗ Model is None after loading")
                logger.error(f"[Inpainting Load] ModelManager type: {type(self.model_manager)}")
                logger.error(f"[Inpainting Load] ModelManager dir: {dir(self.model_manager)}")
                
        except ImportError as e:
            logger.error(f"[Inpainting Load] ✗ Import error: {e}")
            logger.error(f"[Inpainting Load] Please install: pip install lama-cleaner torch torchvision")
            import traceback
            logger.error(traceback.format_exc())
            self.model = None
            self.model_manager = None
        except Exception as e:
            logger.error(f"[Inpainting Load] ✗ Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.model = None
            self.model_manager = None
        
        logger.info(f"[Inpainting Load] ════════════════════════════════════════")
    
    async def inpaint_text_regions(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        dilate_kernel_size: int = 5
    ) -> Optional[np.ndarray]:
        """
        Inpaint text regions using LaMa model
        
        Args:
            image: Input image (RGB, H x W x 3)
            mask: Binary mask của text regions (H x W), 255 = text, 0 = background
            dilate_kernel_size: Kernel size để dilate mask (default: 5)
        
        Returns:
            np.ndarray: Inpainted image (RGB) hoặc None nếu thất bại
        """
        logger.info(f"[Inpainting] ════════════════════════════════════════")
        logger.info(f"[Inpainting] Starting inpaint_text_regions()")
        logger.info(f"[Inpainting] Image shape: {image.shape}")
        logger.info(f"[Inpainting] Image dtype: {image.dtype}")
        logger.info(f"[Inpainting] Mask shape: {mask.shape}")
        logger.info(f"[Inpainting] Mask dtype: {mask.dtype}")
        logger.info(f"[Inpainting] Model status: {self.model is not None}")
        logger.info(f"[Inpainting] ModelManager status: {self.model_manager is not None}")
        
        if self.model is None or self.model_manager is None:
            logger.warning("[Inpainting] ✗ Model not loaded, returning original image")
            logger.warning("[Inpainting] Reason: model or model_manager is None")
            return image
        
        try:
            mask_coverage = np.sum(mask > 0) / mask.size * 100
            mask_pixels = np.sum(mask > 0)
            logger.info(f"[Inpainting] Mask coverage: {mask_coverage:.2f}% ({mask_pixels} pixels)")
            
            if mask_coverage < 0.01:
                logger.warning(f"[Inpainting] ✗ Mask coverage too low ({mask_coverage:.2f}%), skipping inpainting")
                return image
            
            # Dilate mask để cover text tốt hơn
            logger.info(f"[Inpainting] Dilating mask with kernel_size={dilate_kernel_size}")
            if dilate_kernel_size > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                dilated_coverage = np.sum(mask_dilated > 0) / mask_dilated.size * 100
                logger.info(f"[Inpainting] After dilation: {dilated_coverage:.2f}% coverage")
            else:
                mask_dilated = mask
            
            # Ensure mask is binary (0 hoặc 255)
            mask_dilated = (mask_dilated > 0).astype(np.uint8) * 255
            logger.info(f"[Inpainting] Mask binary conversion: unique values = {np.unique(mask_dilated)}")
            
            # Convert image to uint8 if needed
            if image.dtype != np.uint8:
                logger.info(f"[Inpainting] Converting image from {image.dtype} to uint8")
                image_input = (image * 255).astype(np.uint8)
            else:
                image_input = image
            
            logger.info(f"[Inpainting] Final input - Image: {image_input.shape} {image_input.dtype}, Mask: {mask_dilated.shape} {mask_dilated.dtype}")
            
            # LaMa expects RGB image và binary mask
            logger.info(f"[Inpainting] Importing Config...")
            from lama_cleaner.schema import Config
            
            # Tạo config cho LaMa inpainting
            logger.info(f"[Inpainting] Creating Config for LaMa...")
            config = Config(
                ldm_steps=25,
                ldm_sampler='plms',
                hd_strategy='Original',
                hd_strategy_crop_margin=196,
                hd_strategy_crop_trigger_size=1280,
                hd_strategy_resize_limit=2048,
                prompt='',
                negative_prompt='',
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
                sd_sampler='ddim',
                sd_seed=42,
                sd_match_histograms=False,
                cv2_flag='INPAINT_NS',
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
                controlnet_method='control_v11p_sd15_canny',
            )
            logger.info(f"[Inpainting] ✓ Config created")
            
            logger.info(f"[Inpainting] Running model_manager() for inpainting...")
            logger.info(f"[Inpainting] This may take 5-30 seconds depending on image size and device...")
            
            import time
            start_time = time.time()
            
            # Run inpainting
            result = self.model_manager(image_input, mask_dilated, config)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"[Inpainting] ✓ Inpainting completed in {elapsed_time:.2f}s")
            logger.info(f"[Inpainting] Result shape: {result.shape if result is not None else 'None'}")
            logger.info(f"[Inpainting] Result dtype: {result.dtype if result is not None else 'None'}")
            
            if result is None:
                logger.error(f"[Inpainting] ✗ Result is None!")
                return image
            
            # Check if result is different from input
            if np.array_equal(result, image_input):
                logger.warning(f"[Inpainting] ⚠ Result is identical to input (no changes)")
            else:
                diff_pixels = np.sum(result != image_input)
                diff_percentage = (diff_pixels / result.size) * 100
                logger.info(f"[Inpainting] ✓ Result differs from input: {diff_pixels} pixels ({diff_percentage:.2f}%)")
            
            logger.info(f"[Inpainting] ════════════════════════════════════════")
            
            return result
            
        except Exception as e:
            logger.error(f"[Inpainting] ✗ Error during inpainting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info(f"[Inpainting] Returning original image due to error")
            return image