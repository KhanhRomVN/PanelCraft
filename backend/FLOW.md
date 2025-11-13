┌─────────────────────────────────────────────────────────────────┐
│                    ĐẦU VÀO: Ảnh gốc (Original Image)            │
│                         (Định dạng RGB)                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│        BƯỚC 1: PHÂN ĐOẠN BUBBLE (Boundary Segmentation)       │
│                  (manga_bubble_seg.onnx)                      │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ├─► Tiền xử lý (Preprocess):
                             │   • Resize về 640x640
                             │   • Thêm padding (letterbox)
                             │   • Chuẩn hóa (normalize) giá trị 0-1
                             │
                             ├─► Chạy inference:
                             │   • Detection boxes: shape (8400, 37)
                             │   • Mask prototypes: shape (1, 32, 160, 160)
                             │
                             ├─► Hậu xử lý (Postprocess):
                             │   • Lọc theo confidence >= 0.5
                             │   • Áp dụng NMS (IoU=0.45)
                             │   • Tạo binary masks từ prototypes
                             │   • Crop masks theo bounding boxes
                             │   • Resize masks về kích thước gốc
                             │
                             ├─► Lọc chất lượng (Quality Filter):
                             │   • Lọc theo std_threshold=2.0
                             │   • Loại bỏ outliers (diện tích, tỷ lệ khung hình)
                             │   • Diện tích tối thiểu >= 100 pixels
                             │
                             └─► Tìm hình chữ nhật nội tiếp (Inscribed Rectangle):
                                 • Tìm hình chữ nhật lớn nhất bên trong mask
                                 • Lưu tọa độ dạng [x, y, w, h]
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KẾT QUẢ BƯỚC 1:                               │
│  • segments: [{id, box, score, rectangle, mask}]                │
│  • visualization: Ảnh với outline xanh lá + hình chữ nhật đỏ   │
│  • masks: List[np.ndarray] - Binary masks cho mỗi segment       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│           BƯỚC 2A: PHÁT HIỆN TẤT CẢ TEXT BOXES                        │
│         (Detect trên ÁNH GỐC - TRƯỚC KHI clean bất kỳ gì)             │
│                  (comictextdetector.pt.onnx)                          │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ├─► Preprocess:
                             │   • Letterbox resize về 1024x1024
                             │   • Normalize 0-1
                             │
                             ├─► Run inference:
                             │   • Detect trên TOÀN BỘ ảnh gốc
                             │   • Output: boxes, mask, lines_map
                             │
                             ├─► Postprocess:
                             │   • Apply NMS (conf=0.4, iou=0.35)
                             │   • Scale boxes về original size
                             │
                             └─► Output:
                                 • all_text_boxes: np.ndarray [x1,y1,x2,y2]
                                 • all_text_scores: np.ndarray
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│             BƯỚC 2B: LÀM SẠCH TEXT TRONG BUBBLES                │
│        (Clean text CHỈ TRONG các bubble regions)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─► Tạo Blank Canvas:
                             │   • Tạo canvas đen (full size)
                             │   • Paste CHỈ balloon regions lên canvas
                             │   • Sử dụng binary masks từ Bước 1
                             │
                             ├─► Text Detection trên Blank Canvas:
                             │   • Letterbox resize về 1024x1024
                             │   • Run comictextdetector.pt.onnx
                             │   • Apply NMS (conf=0.4, iou=0.35)
                             │   • Extract global text_mask (binary)
                             │
                             ├─► Inpaint Text:
                             │   • cv2.inpaint() trên blank canvas
                             │   • Method: INPAINT_TELEA
                             │   • Radius: 5 pixels
                             │
                             └─► Paste Back:
                                 • Copy cleaned regions từ canvas
                                 • Paste vào vị trí segments trên ảnh gốc
                                 • Blend bằng binary masks
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   KẾT QUẢ BƯỚC 2B:                              │
│  • cleaned_image: Ảnh gốc đã clean text TRONG bubbles          │
│  • URL: /temp/cleaned_text_*.jpg                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          BƯỚC 2C: LỌC TEXT BOXES NGOÀI BUBBLES                  │
│       (Filter từ all_text_boxes của Bước 2A)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─► Logic Filter:
                             │   • Loop qua all_text_boxes
                             │   • Tính center (x, y) của mỗi box
                             │   • Check overlap với segments:
                             │     - Nếu center TRONG segment box → BỎ QUA
                             │     - Nếu center NGOÀI tất cả segments → GIỮ LẠI
                             │
                             ├─► Quality Filter (Optional):
                             │   • Lọc theo std_threshold=2.5
                             │   • Min area >= 30 pixels
                             │   • Min score >= 0.2
                             │
                             └─► Output:
                                 • text_boxes_outside: np.ndarray
                                 • text_scores_outside: np.ndarray
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│        BƯỚC 2D: XỬ LÝ TEXT NGOÀI BUBBLES (WITH OCR)            │
│     (Process trên CLEANED_IMAGE thay vì ảnh gốc)               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─► Detect Text trên Cleaned Image:
                             │   • Input: cleaned_image (từ Bước 2B)
                             │   • Run comictextdetector.pt.onnx
                             │   • Extract global_mask từ output
                             │
                             ├─► Tạo Individual Masks:
                             │   • Loop qua text_boxes_outside
                             │   • Crop mask region cho từng box
                             │   • Crop image region cho OCR
                             │
                             ├─► OCR Verification:
                             │   • Run manga_ocr.MangaOcr() trên mỗi region
                             │   • Validate text với _is_valid_text()
                             │   • Check: length > 0, có ký tự meaningful
                             │
                             ├─► Quality Filter (Optional - hiện tắt):
                             │   • Filter by std_threshold=2.0
                             │   • Min area >= 30
                             │   • Chỉ giữ boxes có is_valid=True
                             │
                             └─► Output:
                                 • text_outside_data: List[dict]
                                   - id, box, score, mask
                                   - ocr_text, ocr_confidence
                                   - is_valid (bool)
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               VISUALIZATION - TEXT OUTSIDE BUBBLES              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─► Tạo Visualization Image:
                             │   • Base: cleaned_image
                             │   • Bôi màu ĐỎ (0.5 alpha) lên text masks
                             │   • Vẽ GREEN rectangles (thickness=2)
                             │   • Hiển thị OCR text trên boxes
                             │
                             └─► Save:
                                 • URL: /temp/text_outside_with_masks_*.jpg
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BƯỚC 3: NHẬN DẠNG CHỮ (OCR)                  │
│              (OCR cho text TRONG bubble segments)               │
│                    (Model: manga-ocr-base)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─► Xử lý từng Segment:
                             │   • Crop vùng ảnh theo segment.box
                             │   • Convert sang PIL Image
                             │   • Run manga_ocr.MangaOcr()
                             │   • Extract text string
                             │
                             └─► Output:
                                 • segment_id
                                 • original_text
                                 • confidence=1.0 (default)
                                 
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KẾT QUẢ CUỐI CÙNG                           │
│                  (Đối tượng ImageResult)                        │
└─────────────────────────────────────────────────────────────────┘
│                                                                   │
│  • original_path: str                                            │
│  • original_dimensions: (width, height)                          │
│                                                                   │
│  • cleaned_text_result: str                                      │
│    - URL của ảnh đã clean text TRONG bubbles                    │
│    - Format: /temp/cleaned_text_*.jpg                           │
│                                                                   │
│  • segments: List[SegmentData]                                   │
│    - id, box [x1,y1,x2,y2], score                               │
│    - rectangle [x,y,w,h] (largest inscribed)                    │
│                                                                   │
│  • rectangles: List[dict]                                        │
│    - Metadata cho Frontend: {id, x, y, w, h}                    │
│                                                                   │
│  • ocr_results: List[OCRResult]                                  │
│    - segment_id, original_text, confidence                       │
│    - OCR cho text TRONG bubbles                                 │
│                                                                   │
│  • text_outside_bubbles: List[dict]                              │
│    - id, box [x1,y1,x2,y2], score                               │
│    - mask: np.ndarray (individual binary mask)                  │
│    - ocr_text: str (từ manga-ocr)                               │
│    - ocr_confidence: float                                       │
│    - is_valid: bool (text quality check)                        │
│                                                                   │
│  • VISUALIZATION URLs:                                           │
│    - Segmentation: /temp/visualization_*.jpg                     │
│      (Green outlines + Red rectangles)                           │
│    - Text Outside: /temp/text_outside_with_masks_*.jpg          │
│      (Red masks + Green boxes + OCR text)                       │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘