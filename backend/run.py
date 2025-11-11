import sys
import uvicorn
from app.main import app
from app.core.config import settings

def is_frozen():
    """Kiểm tra xem có đang chạy từ PyInstaller binary không"""
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

if __name__ == "__main__":
    # Tắt reload khi chạy từ binary để tránh lỗi KeyboardInterrupt
    reload_mode = settings.DEBUG and not is_frozen()
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=reload_mode,
        log_level="info"
    )