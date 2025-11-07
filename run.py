#!/usr/bin/env python3
"""
Script chạy ứng dụng PySide6
"""

import sys
import os

# Thêm thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    sys.exit(main())