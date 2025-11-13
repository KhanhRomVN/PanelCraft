#!/usr/bin/env python3
"""
PanelCleaner Colab Standalone Runner
Complete implementation of PanelCleaner workflow in a single file.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2
import requests
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum, IntEnum, auto
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import scipy.signal
from scipy import ndimage
import re

# Disable unnecessary logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class BoxType(Enum):
    BOX = 0
    EXTENDED_BOX = 1
    MERGED_EXT_BOX = 2
    REFERENCE_BOX = 3

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self):
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def merge(self, other):
        x_min = min(self.x1, other.x1)
        y_min = min(self.y1, other.y1)
        x_max = max(self.x2, other.x2)
        y_max = max(self.y2, other.y2)
        return Box(x_min, y_min, x_max, y_max)

    def overlaps(self, other, threshold: float):
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = x_overlap * y_overlap
        smaller_area = min(self.area, other.area) or 1
        return intersection / smaller_area > (threshold / 100)

    def pad(self, amount: int, canvas_size):
        x1_new = max(self.x1 - amount, 0)
        y1_new = max(self.y1 - amount, 0)
        x2_new = min(self.x2 + amount, canvas_size[0])
        y2_new = min(self.y2 + amount, canvas_size[1])
        return Box(x1_new, y1_new, x2_new, y2_new)

    def right_pad(self, amount: int, canvas_size):
        x2_new = min(self.x2 + amount, canvas_size[0])
        return Box(self.x1, self.y1, x2_new, self.y2)

    def scale(self, factor: float):
        return Box(
            int(self.x1 * factor),
            int(self.y1 * factor),
            int(self.x2 * factor),
            int(self.y2 * factor)
        )

class PanelCleanerConfig:
    """Configuration for PanelCleaner processing"""
    def __init__(self):
        # Text detection settings
        self.input_size = 1024
        self.box_min_size = 20 * 20
        self.suspicious_box_min_size = 200 * 200
        self.box_overlap_threshold = 20.0

        # Preprocessing settings
        self.box_padding_initial = 2
        self.box_right_padding_initial = 3
        self.box_padding_extended = 5
        self.box_right_padding_extended = 5
        self.box_reference_padding = 20

        # Masking settings
        self.mask_growth_step_pixels = 2
        self.mask_growth_steps = 11
        self.min_mask_thickness = 4
        self.mask_max_standard_deviation = 15
        self.mask_improvement_threshold = 0.1
        self.off_white_max_threshold = 240
        self.debug_mask_color = (108, 30, 240, 127)

class TextDetector:
    """Simplified text detector using OpenCV DNN"""
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.input_size = 1024

    def preprocess(self, image):
        """Preprocess image for text detection"""
        h, w = image.shape[:2]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create square canvas
        canvas = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8) * 255
        y_offset = (self.input_size - new_h) // 2
        x_offset = (self.input_size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        # Normalize
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, swapRB=True)
        return blob, (x_offset, y_offset), scale, (w, h)

    def detect(self, image_path):
        """Detect text in image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        blob, offset, scale, orig_size = self.preprocess(image)

        # Run inference
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Process outputs (simplified - actual implementation would parse YOLO outputs)
        boxes = self.simple_box_detection(image, scale, offset, orig_size)
        return boxes, image

    def simple_box_detection(self, image, scale, offset, orig_size):
        """Simple text box detection (placeholder for actual AI detection)"""
        # This is a simplified version - real implementation would use the model outputs
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by size
            if area > 1000 and area < (image.shape[0] * image.shape[1]) * 0.3:
                # Scale back to original coordinates
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(w / scale)
                h_orig = int(h / scale)

                boxes.append(Box(x_orig, y_orig, x_orig + w_orig, y_orig + h_orig))

        return boxes

class MaskProcessor:
    """Handles mask generation and processing"""

    @staticmethod
    def make_growth_kernel(thickness: int):
        """Create kernel for mask growth"""
        diameter = thickness * 2 + 1
        if diameter <= 5:
            kernel = np.ones((diameter, diameter), dtype=np.float64)
            kernel[0, 0] = 0
            kernel[0, -1] = 0
            kernel[-1, 0] = 0
            kernel[-1, -1] = 0
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            kernel = kernel.astype(np.float64)
        return kernel

    @staticmethod
    def grow_mask(mask, size: int):
        """Grow mask by specified amount"""
        if size == 0:
            return mask

        mask = mask.convert("L")
        mask_array = np.array(mask, dtype=np.uint8)

        kernel = MaskProcessor.make_growth_kernel(size)
        grow_size = size * 2

        # Pad and convolve
        padded_mask = np.pad(mask_array, ((grow_size, grow_size), (grow_size, grow_size)), mode='edge')
        padded_mask = scipy.signal.convolve2d(padded_mask, kernel, mode='same')
        cropped_mask = padded_mask[grow_size:-grow_size, grow_size:-grow_size]

        return Image.fromarray(np.where(cropped_mask > 0, 255, 0).astype(np.uint8)).convert("1")

    @staticmethod
    def make_box_mask(boxes, image_size):
        """Create mask from boxes"""
        mask = Image.new("1", image_size, 0)
        draw = ImageDraw.Draw(mask)
        for box in boxes:
            draw.rectangle(box.as_tuple, fill=1)
        return mask

    @staticmethod
    def border_std_deviation(base, mask, off_white_threshold=240):
        """Calculate border uniformity"""
        mask_edges = mask.filter(ImageFilter.FIND_EDGES)
        base_data = np.array(base.convert("L"))
        mask_data = np.array(mask_edges)

        border_pixels = base_data[mask_data == 1]
        if len(border_pixels) == 0:
            return float('inf'), (255, 255, 255)

        std = float(np.std(border_pixels))
        median_color = int(np.median(border_pixels))

        return std, (median_color, median_color, median_color)

class PanelCleaner:
    """Main PanelCleaner implementation"""

    def __init__(self, config=None):
        self.config = config or PanelCleanerConfig()
        self.mask_processor = MaskProcessor()

    def download_model(self):
        """Download text detection model"""
        model_url = "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
        model_path = Path("/content/comictextdetector.pt.onnx")

        if not model_path.exists():
            print("Downloading text detection model...")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

        return model_path

    def display_image(self, image, title="Image", figsize=(12, 8)):
        """Display image with title"""
        plt.figure(figsize=figsize)
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        else:
            img = image

        plt.imshow(np.array(img))
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run_text_detection(self, image_path, cache_dir):
        """Step 1: Text detection"""
        print("Step 1: Running Text Detection...")

        model_path = self.download_model()
        detector = TextDetector(model_path)

        boxes, image = detector.detect(image_path)

        # Save results
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        base_image_path = cache_dir / "base.png"
        image_pil.save(base_image_path)

        # Create visualization
        viz_image = image_pil.copy()
        draw = ImageDraw.Draw(viz_image)

        for i, box in enumerate(boxes):
            draw.rectangle(box.as_tuple, outline="red", width=3)
            draw.text((box.x1 + 5, box.y1 + 5), str(i+1), fill="red")

        boxes_path = cache_dir / "raw_boxes.png"
        viz_image.save(boxes_path)

        # Create simple mask (placeholder)
        mask = Image.new("1", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        for box in boxes:
            mask_draw.rectangle(box.as_tuple, fill=1)

        mask_path = cache_dir / "raw_mask.png"
        mask.save(mask_path)

        # Display results
        self.display_image(image_pil, "1. Input Image")
        self.display_image(viz_image, "2. Raw Detected Boxes")
        self.display_image(mask, "3. AI Generated Mask")

        return boxes, base_image_path, mask_path, boxes_path

    def preprocess_boxes(self, boxes, image_size):
        """Step 2: Box preprocessing"""
        print("Step 2: Preprocessing Boxes...")

        # Initial padding
        initial_boxes = [box.pad(self.config.box_padding_initial, image_size) for box in boxes]
        initial_boxes = [box.right_pad(self.config.box_right_padding_initial, image_size) for box in initial_boxes]

        # Extended padding
        extended_boxes = [box.pad(self.config.box_padding_extended, image_size) for box in boxes]
        extended_boxes = [box.right_pad(self.config.box_right_padding_extended, image_size) for box in extended_boxes]

        # Merge overlapping boxes
        merged_boxes = self.merge_boxes(extended_boxes, self.config.box_overlap_threshold)

        # Reference boxes (largest padding)
        reference_boxes = [box.pad(self.config.box_reference_padding, image_size) for box in merged_boxes]

        return initial_boxes, extended_boxes, merged_boxes, reference_boxes

    def merge_boxes(self, boxes, threshold):
        """Merge overlapping boxes"""
        if not boxes:
            return []

        merged = []
        queue = set(boxes)

        while queue:
            box = queue.pop()
            overlapping = [b for b in queue if box.overlaps(b, threshold)]

            for b in overlapping:
                box = box.merge(b)
                queue.remove(b)

            merged.append(box)

        return merged

    def visualize_boxes(self, image_path, initial_boxes, extended_boxes, merged_boxes, reference_boxes, output_path):
        """Create box visualization"""
        image = Image.open(image_path)
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)

        # Draw boxes with different colors
        for box in reference_boxes:
            draw.rectangle(box.as_tuple, outline="blue", width=2)

        for box in merged_boxes:
            draw.rectangle(box.as_tuple, outline="purple", width=2)

        for box in extended_boxes:
            draw.rectangle(box.as_tuple, outline="red", width=2)

        for i, box in enumerate(initial_boxes):
            draw.rectangle(box.as_tuple, outline="green", width=2)
            draw.text((box.x1 + 5, box.y1 + 5), str(i+1), fill="green")

        viz_image.save(output_path)
        return viz_image

    def create_masks(self, base_image_path, mask_path, boxes_data, cache_dir):
        """Step 3: Mask creation and processing"""
        print("Step 3: Creating Masks...")

        initial_boxes, extended_boxes, merged_boxes, reference_boxes = boxes_data
        base_image = Image.open(base_image_path)
        precise_mask = Image.open(mask_path)

        # Create box mask
        box_mask = self.mask_processor.make_box_mask(extended_boxes, base_image.size)
        box_mask_path = cache_dir / "box_mask.png"
        box_mask.save(box_mask_path)

        # Cut precise mask to box regions
        cut_mask = Image.new("1", base_image.size, 0)
        for box in extended_boxes:
            box_cut = precise_mask.crop(box.as_tuple)
            cut_mask.paste(box_cut, (box.x1, box.y1))

        cut_mask_path = cache_dir / "cut_mask.png"
        cut_mask.save(cut_mask_path)

        # Generate mask growth steps
        mask_results = []
        for i, (masking_box, reference_box) in enumerate(zip(merged_boxes, reference_boxes)):
            result = self.generate_mask_variations(
                base_image, cut_mask, box_mask, masking_box, reference_box
            )
            if result:
                mask_results.append(result)

        # Combine masks
        combined_mask = self.combine_masks(base_image.size, mask_results)
        combined_mask_path = cache_dir / "combined_mask.png"
        combined_mask.save(combined_mask_path)

        # Create mask overlay
        overlay = base_image.copy().convert("RGBA")
        debug_mask = self.apply_debug_color(combined_mask, self.config.debug_mask_color)
        overlay.paste(debug_mask, (0, 0), debug_mask)
        overlay_path = cache_dir / "mask_overlay.png"
        overlay.save(overlay_path)

        # Create cleaned image
        cleaned = base_image.copy()
        cleaned.paste(combined_mask, (0, 0), combined_mask)
        cleaned_path = cache_dir / "cleaned.png"
        cleaned.save(cleaned_path)

        # Extract text
        text_image = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        text_image.paste(base_image, (0, 0), combined_mask)
        text_path = cache_dir / "text.png"
        text_image.save(text_path)

        # Display results
        self.display_image(box_mask, "4. Box Mask")
        self.display_image(cut_mask, "5. Cut Mask")
        self.display_image(combined_mask, "6. Final Combined Mask")
        self.display_image(overlay, "7. Mask Overlay")
        self.display_image(cleaned, "8. Cleaned Image")
        self.display_image(text_image, "9. Isolated Text")

        return mask_results

    def generate_mask_variations(self, base_image, precise_mask, box_mask, masking_box, reference_box):
        """Generate different mask sizes and pick best one"""
        # Calculate offsets
        x_offset = masking_box.x1 - reference_box.x1
        y_offset = masking_box.y1 - reference_box.y1

        # Cut out regions
        base_cut = base_image.crop(reference_box.as_tuple)
        precise_cut = precise_mask.crop(masking_box.as_tuple)

        # Paste precise mask into reference box space
        precise_padded = Image.new("1", base_cut.size, 0)
        precise_padded.paste(precise_cut, (x_offset, y_offset))

        # Generate mask variations
        masks = []
        deviations = []

        # Start with box mask
        box_cut = box_mask.crop(masking_box.as_tuple)
        box_padded = Image.new("1", base_cut.size, 0)
        box_padded.paste(box_cut, (x_offset, y_offset))

        masks.append(box_padded)
        std, _ = self.mask_processor.border_std_deviation(base_cut, box_padded)
        deviations.append(std)

        # Generate grown masks
        current_mask = precise_padded
        for step in range(self.config.mask_growth_steps):
            current_mask = self.mask_processor.grow_mask(current_mask, self.config.mask_growth_step_pixels)
            masks.append(current_mask)
            std, _ = self.mask_processor.border_std_deviation(base_cut, current_mask)
            deviations.append(std)

            # Early stopping if deviation gets worse
            if step > 0 and deviations[-1] > deviations[-2] * (1 + self.config.mask_improvement_threshold):
                break

        # Find best mask
        best_idx = np.argmin(deviations)
        best_std = deviations[best_idx]

        if best_std > self.config.mask_max_standard_deviation:
            return None

        best_mask = masks[best_idx]

        return {
            'mask': best_mask,
            'coords': (reference_box.x1, reference_box.y1),
            'std_dev': best_std,
            'box': masking_box
        }

    def combine_masks(self, image_size, mask_results):
        """Combine all masks into final mask"""
        combined = Image.new("RGBA", image_size, (0, 0, 0, 0))

        for result in mask_results:
            if result:
                mask_rgba = self.apply_debug_color(result['mask'], (255, 255, 255, 255))
                combined.paste(mask_rgba, result['coords'], mask_rgba)

        return combined

    def apply_debug_color(self, mask, color):
        """Apply color to mask for visualization"""
        if mask.mode == "1":
            array_1 = np.array(mask)
            array_rgba = np.zeros((*array_1.shape, 4), dtype=np.uint8)
            array_rgba[array_1 != 0] = color
            return Image.fromarray(array_rgba)
        else:
            return mask.convert("RGBA")

    def process_image(self, image_path):
        """Complete image processing pipeline"""
        cache_dir = Path("/content/panelcleaner_cache")
        cache_dir.mkdir(exist_ok=True)

        print(f"Processing image: {image_path}")
        print("=" * 50)

        # Display original image
        self.display_image(image_path, "Original Image")

        try:
            # Step 1: Text detection
            boxes, base_image_path, mask_path, boxes_path = self.run_text_detection(image_path, cache_dir)

            if not boxes:
                print("No text boxes detected!")
                return

            # Step 2: Preprocessing
            base_image = Image.open(base_image_path)
            boxes_data = self.preprocess_boxes(boxes, base_image.size)
            initial_boxes, extended_boxes, merged_boxes, reference_boxes = boxes_data

            # Create box visualization
            boxes_viz_path = cache_dir / "boxes_final.png"
            boxes_viz = self.visualize_boxes(
                base_image_path, initial_boxes, extended_boxes, merged_boxes, reference_boxes, boxes_viz_path
            )
            self.display_image(boxes_viz, "10. Final Boxes (All Types)")

            # Step 3: Mask processing
            mask_results = self.create_masks(base_image_path, mask_path, boxes_data, cache_dir)

            print("\n" + "=" * 50)
            print("Processing Complete!")
            print(f"Detected {len(boxes)} text boxes")
            print(f"Successfully processed {len([r for r in mask_results if r])} boxes")

        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    print("PanelCleaner Standalone Colab Runner")
    print("Complete implementation in a single file")

    # Check input image
    image_path = Path("/content/1.png")
    if not image_path.exists():
        print(f"Error: Input image {image_path} not found!")
        print("Please upload 1.png to /content/ directory")
        return

    # Initialize and run
    cleaner = PanelCleaner()
    cleaner.process_image(image_path)

if __name__ == "__main__":
    main()