"""Constants for home page and processing pipeline"""

# ========== SEGMENTATION CONSTANTS ==========
SEGMENTATION_INPUT_SIZE = 640
SEGMENTATION_CONF_THRESHOLD = 0.5
SEGMENTATION_IOU_THRESHOLD = 0.45
SEGMENTATION_MASK_THRESHOLD = 0.3

# ========== TEXT DETECTION CONSTANTS ==========
TEXT_DETECTION_INPUT_SIZE = 1024
TEXT_DETECTION_CONF_THRESHOLD = 0.4
TEXT_DETECTION_NMS_THRESHOLD = 0.35
TEXT_DETECTION_MASK_THRESHOLD = 0.3

# ========== OCR CONSTANTS ==========
OCR_MAX_TEXT_LENGTH = 1000
OCR_DISPLAY_MAX_LENGTH = 50

# ========== UI CONSTANTS ==========
MIN_CONTROL_PANEL_WIDTH = 300
CANVAS_PANEL_DEFAULT_RATIO = 0.6
CONTROL_PANEL_DEFAULT_RATIO = 0.4

# ========== IMAGE VALIDATION ==========
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MIN_RECTANGLE_SIZE = 10
RESIZE_HANDLE_SIZE = 8

# ========== MODEL FILES ==========
SEGMENTATION_MODEL_FILE = "yolov8_converted.onnx"
TEXT_DETECTION_MODEL_FILE = "comictextdetector.pt.onnx"
OCR_MODEL_FILES = [
    'config.json',
    'preprocessor_config.json',
    'pytorch_model.bin',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'vocab.txt'
]

# ========== PROCESSING ==========
INPAINT_RADIUS = 5
LETTERBOX_COLOR = (0, 0, 0)
LETTERBOX_STRIDE = 64
PADDING_VALUE = 114