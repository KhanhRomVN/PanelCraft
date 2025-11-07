import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class AppConfig:
    """Application configuration"""
    window_width: int = 1200
    window_height: int = 800
    theme: str = "light"
    language: str = "en"
    enable_hotload: bool = True  # Enable hotload in development
    
    CONFIG_FILE = "config.json"
    
    def load(self) -> bool:
        """Load configuration from file"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
            
    def save(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(asdict(self), f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False