# PanelCraft Backend Refactor Plan (Updated)

Mục tiêu: Chuẩn hóa cấu trúc thư mục để:
- Phân tách rõ infrastructure (core) vs business logic (domains)
- Gom cấu hình tập trung (config)
- Làm rõ layer API (api/routes + api/schemas)
- Tái sử dụng tiện ích (shared)
- Giữ nguyên hành vi hiện tại (không thay đổi logic) trong giai đoạn 1
- Cung cấp backward compatibility tạm thời cho các import cũ (shim / deprecation warnings)

## 1. Cấu trúc MỚI (đích)

```
app/
├── main.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── constants.py
│   └── logging.py
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── processing.py
│   └── schemas/
│       ├── __init__.py
│       └── pipeline.py
├── domains/
│   ├── __init__.py
│   ├── image_processing/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   ├── segmentation_service.py
│   │   ├── text_detection_service.py
│   │   ├── ocr_service.py
│   │   ├── inpainting_service.py
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   └── steps/
│   │   │       ├── __init__.py
│   │   │       └── base.py
│   └── batch_processing/
│       ├── __init__.py
│       ├── store.py (planned – hiện tại dùng trực tiếp state/batch_store.py)
├── core/
│   ├── __init__.py
│   ├── ml_models/
│   │   ├── __init__.py
│   │   ├── base.py
│   ├── image_utils/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   └── processing.py
├── shared/
│   ├── __init__.py
│   ├── exceptions.py (planned)
│   ├── types.py (planned)
│   └── utils.py (planned)
├── services/ (LEGACY SHIMS)
│   ├── pipeline_service.py
│   ├── segmentation_service.py
│   ├── text_detection_service.py
│   ├── ocr_service.py
│   ├── inpainting_service.py
├── utils/ (LEGACY SHIMS)
│   ├── geometry_utils.py
│   └── image_utils.py
├── legacy/ (planned final stage – hiện chưa tạo vì đã dùng shim approach)
```

Giai đoạn 1 (Hoàn tất): Tổ chức lại, tạo domain modules & shim deprecation.
Giai đoạn 2 (Trong tiến trình): Chuẩn hóa import, chuẩn hóa exception & schema usage.
Giai đoạn 3 (Chưa bắt đầu): Xóa shim / dọn sạch legacy.

## 2. Mapping Chi Tiết (Cập nhật trạng thái)

| Hiện tại (cũ)                           | Mới                                    | Trạng thái |
|----------------------------------------|----------------------------------------|-----------|
| app/core/config.py                     | app/config/settings.py                 | Hoàn tất |
| app/core/constants.py                  | app/config/constants.py                | Hoàn tất |
| app/core/logging_config.py             | app/config/logging.py                  | Hoàn tất |
| app/api/routers/processing.py          | app/api/routes/processing.py           | Hoàn tất |
| app/schemas/pipeline.py                | app/api/schemas/pipeline.py            | Không đổi |
| app/services/pipeline_service.py       | app/domains/image_processing/service.py| Refactor + shim |
| app/services/segmentation_service.py   | app/domains/image_processing/segmentation_service.py | Refactor + shim |
| app/services/text_detection_service.py | app/domains/image_processing/text_detection_service.py | Refactor + shim |
| app/services/ocr_service.py            | app/domains/image_processing/ocr_service.py | Refactor + shim |
| app/services/inpainting_service.py     | app/domains/image_processing/inpainting_service.py | Refactor + shim |
| app/utils/geometry_utils.py            | app/core/image_utils/geometry.py       | Consolidated + shim |
| app/utils/image_utils.py               | app/core/image_utils/processing.py     | Consolidated + shim |
| app/state/batch_store.py               | (planned) app/domains/batch_processing/store.py | CHƯA di chuyển |
| app/api/endpoints.py                   | legacy api_endpoints (planned)         | Chưa – consider removal / convert to router or delete |
| app/api/models.py                      | app/api/schemas/... (re-export)        | Deprecated (was duplicate) |
| app/models/pipeline_models.py          | (Removed after ensuring schemas central) | Pending removal audit |

## 3. Bước Đã Hoàn Thành

- Tạo domain folder & di chuyển toàn bộ service logic.
- Thêm docstrings + type hints cho: segmentation, text detection, OCR, inpainting, pipeline orchestrator.
- Chuẩn hóa logging prefix & pattern.
- Thêm degraded mode cho OCR & Inpainting.
- Tách image_utils & geometry_utils vào core/image_utils với authoritative module.
- Tạo shims deprecation cho tất cả legacy services & utils.
- Tạo pipeline step abstraction (base.py) cho mở rộng tương lai.

## 4. Bước Tiếp Theo

1. Batch Processing:
   - Di chuyển app/state/batch_store.py -> app/domains/batch_processing/store.py
   - Tạo thin wrapper `service.py` nếu cần (start/stop batch job patterns).
2. Exception Handling Standardization:
   - Tạo `shared/exceptions.py` (BaseAppError, DomainError, ValidationError).
   - Thay thế raise ValueError / RuntimeError bằng custom exceptions.
3. Response Schema Standardization:
   - Đảm bảo mọi endpoint trả về PipelineResponse / BatchResponse nhất quán.
   - Thay các dict thô bằng schema (e.g. check_models).
4. Import Cleanup:
   - search_files để thay thế mọi import từ `app.services.*` -> `app.domains.image_processing.*`
   - Tương tự cho `app.utils.*` -> `app.core.image_utils.*`
5. Legacy Removal Criteria:
   - Khi số lượng import legacy trả về 0 (regex), xóa shim files.
6. Shared Layer:
   - Implement shared/types.py (common Pydantic types if needed).
   - Implement shared/utils.py (generic helpers).
7. README / Architecture Section:
   - Thêm mô tả pipeline architecture & layering.
8. Performance / Concurrency (Optional Phase 2):
   - Thêm asyncio.gather cho OCR / text detection segment loops (bounded semaphore).
9. Metrics / Observability (Optional):
   - Hook timings (StepResult.duration) vào Prometheus exporter.

## 5. Import Replacement Regex (Chuẩn bị)

Ví dụ dùng `search_files` rồi `replace_in_file`:
- Pattern legacy services: `from\s+app\.services\.(\w+_service)\s+import`
- Replace template: `from app.domains.image_processing.\1 import`

## 6. Rủi Ro & Giải Pháp (Update)

| Rủi ro | Trạng thái | Giải pháp |
|--------|-----------|-----------|
| ImportError do refactor sâu | Đã giảm | Giữ shim deprecation tới khi search confirm sạch |
| Duplicate utils gây nhầm lẫn | Đã xử lý | Chỉ core/image_utils/* authoritative |
| Exception không đồng nhất | Còn tồn tại | Tạo shared/exceptions.py, audit raises |
| Endpoint trả về dict raw | Còn | Chuẩn hóa schemas cho mọi response |
| Kích thước context lớn khi debug | Giảm dần | Tách logs chuẩn & giảm noise trong production |

## 7. Checklist Chi Tiết (Hiện Tại)

- [x] Domain folder scaffolding
- [x] Service refactors (segmentation, text detection, OCR, inpainting)
- [x] Pipeline orchestrator refactor
- [x] Logging centralization
- [x] Constants centralization
- [x] Image utils consolidation
- [x] Geometry utils consolidation
- [x] Shims for legacy services
- [x] Shims for legacy utils
- [x] Pipeline step abstractions (base.py)
- [ ] Batch store relocation
- [ ] Exception standardization (shared/exceptions.py)
- [ ] Response schema standardization
- [ ] Legacy import cleanup (search & replace)
- [ ] Remove shim files post-migration
- [ ] README architecture updates
- [ ] Final legacy directory cleanup (optional)
- [ ] Performance concurrency enhancements (phase 2 optional)
- [ ] Metrics / observability hooks (phase 2 optional)
- [ ] Final summary & changelog

## 8. Kế Hoạch Dọn Dẹp Cuối (Final Cleanup Procedure)

1. Run regex searches:
   - `from app\.services\.` → expect 0 matches.
   - `from app\.utils\.geometry_utils` / `from app\.utils\.image_utils` → expect 0.
2. Remove shim files.
3. Confirm imports succeed with `uvicorn app.main:app --reload`.
4. Update README & CHANGELOG.
5. Tag git release (e.g. v1.0-refactor).

## 9. Sample Migration Command (Gợi Ý)

```bash
# Example (dry-run style conceptual, not executed automatically here)
grep -R "from app.services" -n app | cut -d: -f1 | xargs sed -i 's/from app.services./from app.domains.image_processing./g'
```

(Thực tế nên làm thủ công với replace_in_file để tránh lỗi ngoài ý muốn.)

## 10. Summary

Refactor giai đoạn đầu đã hoàn thành: kiến trúc phân tầng rõ ràng, domain services tách riêng, utils hợp nhất, pipeline orchestrator sạch hơn, chuẩn bị nền tảng cho bước mở rộng (steps, concurrency, exceptions). Các shim deprecation đảm bảo backward compatibility trong thời gian chuyển đổi. Các bước còn lại tập trung vào chuẩn hóa exceptions, responses và loại bỏ mã legacy.

---
Generated & updated during iterative refactor session.
