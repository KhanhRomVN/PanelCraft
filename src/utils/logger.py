import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter với màu sắc cho terminal"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Màu cho các thành phần khác
    TIMESTAMP_COLOR = '\033[90m'  # Gray
    NAME_COLOR = '\033[94m'        # Blue
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        """
        Khởi tạo formatter
        
        Args:
            fmt: Format string cho log message
            datefmt: Format string cho timestamp
            use_colors: Có sử dụng màu sắc hay không
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        """Format log record với màu sắc"""
        if not self.use_colors:
            return super().format(record)
        
        # Lưu giá trị gốc
        levelname_orig = record.levelname
        name_orig = record.name
        msg_orig = record.msg
        
        # Thêm màu cho level
        levelname_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{levelname_color}{self.BOLD}{record.levelname:8s}{self.RESET}"
        
        # Thêm màu cho logger name
        record.name = f"{self.NAME_COLOR}{record.name}{self.RESET}"
        
        # Format message
        formatted = super().format(record)
        
        # Khôi phục giá trị gốc
        record.levelname = levelname_orig
        record.name = name_orig
        record.msg = msg_orig
        
        return formatted


class Logger:
    """Class quản lý logger với màu sắc"""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    @staticmethod
    def setup_logger(
        name: str = 'app',
        level: int = logging.INFO,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        use_colors: bool = True,
        log_file: Optional[str] = None
    ) -> logging.Logger:
        """
        Setup logger với màu sắc
        
        Args:
            name: Tên của logger
            level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Custom format string
            date_format: Custom date format string
            use_colors: Có sử dụng màu sắc cho console hay không
            log_file: Đường dẫn file log (optional)
        
        Returns:
            Logger đã được config
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Xóa handlers cũ nếu có
        logger.handlers.clear()
        
        # Format strings
        fmt = log_format or Logger.DEFAULT_FORMAT
        datefmt = date_format or Logger.DEFAULT_DATE_FORMAT
        
        # Console handler với màu
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(fmt, datefmt, use_colors=use_colors)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (không màu)
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(fmt, datefmt)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Ngăn log propagate lên parent logger
        logger.propagate = False
        
        return logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Lấy logger đã tồn tại hoặc tạo mới với config mặc định
        
        Args:
            name: Tên logger
            
        Returns:
            Logger instance
        """
        logger = logging.getLogger(name)
        
        # Nếu chưa có handlers, setup với config mặc định
        if not logger.handlers:
            return Logger.setup_logger(name)
        
        return logger


# Convenience functions
def setup_logger(
    name: str = 'app',
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_colors: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Shortcut function để setup logger"""
    return Logger.setup_logger(name, level, log_format, date_format, use_colors, log_file)


def get_logger(name: str) -> logging.Logger:
    """Shortcut function để lấy logger"""
    return Logger.get_logger(name)


# Demo usage
if __name__ == "__main__":
    # Setup logger với config mặc định
    logger = setup_logger('demo', level=logging.DEBUG)
    
    # Test các level
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print("\n" + "="*80 + "\n")
    
    # Setup logger với custom format và file output
    custom_logger = setup_logger(
        name='custom',
        level=logging.DEBUG,
        log_format='[%(levelname)s] %(name)s: %(message)s',
        log_file='app.log'
    )
    
    custom_logger.info("Custom format logger")
    custom_logger.warning("This also saves to app.log")
    
    print("\n" + "="*80 + "\n")
    
    # Logger cho module cụ thể
    module_logger = setup_logger('myapp.database', level=logging.DEBUG)
    module_logger.debug("Database connection established")
    module_logger.info("Query executed successfully")