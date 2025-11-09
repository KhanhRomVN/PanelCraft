import json
import os
import logging
from typing import List, Optional
from pathlib import Path


class FontManager:
    """Quản lý font settings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings_file = self.get_settings_path()
        self.settings = self.load_settings()
    
    def get_settings_path(self) -> str:
        """Lấy đường dẫn file settings"""
        app_data_dir = Path.home() / ".panelcraft"
        app_data_dir.mkdir(exist_ok=True)
        
        return str(app_data_dir / "font_settings.json")
    
    def load_settings(self) -> dict:
        """Load font settings từ file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load font settings: {e}")
        
        # Default settings
        return {
            'visible_fonts': [
                'Arial',
                'Times New Roman',
                'Courier New',
                'Verdana',
                'Georgia'
            ],
            'default_font': None  # None = "Default (System)"
        }
    
    def save_settings(self) -> bool:
        """Lưu font settings vào file"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save font settings: {e}") 
            return False
    
    def get_visible_fonts(self) -> List[str]:
        """Lấy danh sách fonts hiển thị"""
        fonts = self.settings.get('visible_fonts', [])
        self.logger.info(f"[FONT_MGR] get_visible_fonts() returning {len(fonts)} fonts: {fonts[:3]}..." if len(fonts) > 3 else f"[FONT_MGR] get_visible_fonts() returning {len(fonts)} fonts: {fonts}")
        return fonts
    
    def set_visible_fonts(self, fonts: List[str]) -> bool:
        """Set danh sách fonts hiển thị"""
        self.settings['visible_fonts'] = fonts
        return self.save_settings()
    
    def get_default_font(self) -> Optional[str]:
        """Lấy font mặc định (None = Default)"""
        default = self.settings.get('default_font')
        self.logger.info(f"[FONT_MGR] get_default_font() returning: '{default}'")
        return default
    
    def set_default_font(self, font: Optional[str]) -> bool:
        """Set font mặc định (None = Default)"""
        self.settings['default_font'] = font
        return self.save_settings()