import sys
import os
import logging

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app.application import Application

def main():
    """Entry point chính của ứng dụng"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Khởi tạo và chạy ứng dụng
    app = Application(sys.argv)
    
    if app.initialize():
        return app.exec()
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())