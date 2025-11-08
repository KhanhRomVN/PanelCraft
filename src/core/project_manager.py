import json
import os
import uuid
import logging
from typing import List, Dict, Optional


class ProjectManager:
    """Quản lý manga project và characters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_file = self.get_project_file_path()
        self.project_data = self.load_project()
    
    def get_project_file_path(self) -> str:
        """Lấy đường dẫn file project"""
        # Lưu trong thư mục user data
        from pathlib import Path
        
        app_data_dir = Path.home() / ".panelcraft"
        app_data_dir.mkdir(exist_ok=True)
        
        return str(app_data_dir / "current_project.json")
    
    def load_project(self) -> Dict:
        """Load project data từ file"""
        if os.path.exists(self.project_file):
            try:
                with open(self.project_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load project: {e}")
        
        # Default project structure
        return {
            'id': str(uuid.uuid4()),
            'name': 'Default Project',
            'characters': []
        }
    
    def save_project(self) -> bool:
        """Lưu project data vào file"""
        try:
            with open(self.project_file, 'w', encoding='utf-8') as f:
                json.dump(self.project_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save project: {e}")
            return False
    
    def get_characters(self) -> List[Dict]:
        """Lấy danh sách characters"""
        return self.project_data.get('characters', [])
    
    def add_character(self, name: str) -> bool:
        """Thêm character mới"""
        char_id = f"char_{uuid.uuid4().hex[:8]}"
        
        new_character = {
            'id': char_id,
            'name': name
        }
        
        self.project_data['characters'].append(new_character)
        return self.save_project()
    
    def delete_character(self, char_id: str) -> bool:
        """Xóa character theo ID"""
        self.project_data['characters'] = [
            c for c in self.project_data['characters'] 
            if c['id'] != char_id
        ]
        return self.save_project()
    
    def get_character_by_id(self, char_id: str) -> Optional[Dict]:
        """Lấy character theo ID"""
        for char in self.project_data['characters']:
            if char['id'] == char_id:
                return char
        return None