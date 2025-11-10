import sys
import os
import logging
import signal

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup colored logging cho toàn bộ app
from utils.logger import ColoredFormatter

# Apply colored formatter cho root logger
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    use_colors=True
))
logging.root.handlers = []
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

from app.application import Application

def main():
    """Entry point chính của ứng dụng"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Khởi tạo ứng dụng
    app = Application(sys.argv)
    
    # Setup signal handler cho Ctrl+C
    def signal_handler(signum, frame):
        """Xử lý tín hiệu Ctrl+C"""
        logging.info("Nhận tín hiệu ngắt (Ctrl+C), đang tắt ứng dụng...")
        app.quit()
    
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Cho phép Python xử lý signal trong Qt event loop
    # Tạo timer để Python có cơ hội xử lý signals
    from PySide6.QtCore import QTimer
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Dummy slot
    timer.start(500)  # Check mỗi 500ms
    
    if app.initialize():
        return app.exec()
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())