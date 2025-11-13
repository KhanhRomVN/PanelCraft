import logging
import sys

def setup_logging():
    """Setup logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(levelname)s:     %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Setup app loggers
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    
    return root_logger