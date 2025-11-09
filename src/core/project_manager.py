import json
import os
import uuid
import logging
from typing import List, Dict, Optional


class ProjectManager:
    """Quản lý manga project và characters"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.projects_db_file = self.get_projects_db_path()
        self.current_project_file = self.get_current_project_path()
        self.project_data = self.load_project()
    
    def get_projects_db_path(self) -> str:
        """Lấy đường dẫn file database chứa tất cả projects"""
        from pathlib import Path
        
        app_data_dir = Path.home() / ".panelcraft"
        app_data_dir.mkdir(exist_ok=True)
        
        return str(app_data_dir / "projects_db.json")
    
    def get_current_project_path(self) -> str:
        """Lấy đường dẫn file lưu current project ID"""
        from pathlib import Path
        
        app_data_dir = Path.home() / ".panelcraft"
        app_data_dir.mkdir(exist_ok=True)
        
        return str(app_data_dir / "current_project.json")
    
    def load_project(self) -> Dict:
        """Load current project data"""
        # Load current project ID
        current_id = self._load_current_project_id()
        
        if not current_id:
            self.logger.info("No current project set")
            return {}
        
        # Load project from database
        projects_db = self._load_projects_db()
        
        if current_id in projects_db:
            return projects_db[current_id]
        
        self.logger.warning(f"Current project {current_id} not found in database")
        return {}
    
    def save_project(self) -> bool:
        """Lưu current project vào database"""
        if not self.project_data or 'id' not in self.project_data:
            self.logger.error("Cannot save: Invalid project data")
            return False
        
        try:
            # Load existing database
            projects_db = self._load_projects_db()
            
            # Update/add current project
            project_id = self.project_data['id']
            projects_db[project_id] = self.project_data
            
            # Save database
            return self._save_projects_db(projects_db)
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
    
    def get_current_project_info(self) -> Optional[dict]:
        """Lấy thông tin project hiện tại"""
        if not self.project_data or self.project_data.get('name') == 'Default Project':
            return None
        
        return {
            'id': self.project_data.get('id'),
            'name': self.project_data.get('name'),
            'author': self.project_data.get('author', ''),
            'description': self.project_data.get('description', ''),
            'created_at': self.project_data.get('created_at', '')
        }
    
    def get_all_projects(self) -> List[Dict]:
        """Lấy danh sách tất cả projects"""
        try:
            projects_db = self._load_projects_db()
            
            # Convert dict to list và sort theo created_at
            projects_list = []
            
            for project_id, project_data in projects_db.items():
                projects_list.append({
                    'id': project_data.get('id'),
                    'name': project_data.get('name'),
                    'author': project_data.get('author', ''),
                    'description': project_data.get('description', ''),
                    'created_at': project_data.get('created_at', '')
                })
            
            # Sort theo created_at (mới nhất trước)
            projects_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return projects_list
        except Exception as e:
            self.logger.error(f"Error getting all projects: {e}")
            return []
        
    def _load_projects_db(self) -> Dict:
        """Load projects database từ file"""
        if os.path.exists(self.projects_db_file):
            try:
                with open(self.projects_db_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load projects database: {e}")
        
        return {}
    
    def _save_projects_db(self, projects_db: Dict) -> bool:
        """Lưu projects database vào file"""
        try:
            with open(self.projects_db_file, 'w', encoding='utf-8') as f:
                json.dump(projects_db, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save projects database: {e}")
            return False
    
    def _load_current_project_id(self) -> Optional[str]:
        """Load current project ID từ file"""
        if os.path.exists(self.current_project_file):
            try:
                with open(self.current_project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('current_project_id')
            except Exception as e:
                self.logger.error(f"Failed to load current project ID: {e}")
        
        return None
    
    def _save_current_project_id(self, project_id: str) -> bool:
        """Lưu current project ID vào file"""
        try:
            with open(self.current_project_file, 'w', encoding='utf-8') as f:
                json.dump({'current_project_id': project_id}, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save current project ID: {e}")
            return False
    
    def _create_project_in_db(self, name: str, author: str, description: str) -> Optional[str]:
        """
        Tạo project mới trong database (internal method)
        
        Returns:
            str: Project ID nếu thành công, None nếu thất bại
        """
        try:
            import datetime
            
            project_id = str(uuid.uuid4())
            created_at = datetime.datetime.now().isoformat()
            
            # Tạo project data structure
            new_project = {
                'id': project_id,
                'name': name,
                'author': author,
                'description': description,
                'created_at': created_at,
                'characters': []
            }
            
            # Load database và add project mới
            projects_db = self._load_projects_db()
            projects_db[project_id] = new_project
            
            # Save database
            if self._save_projects_db(projects_db):
                # Set as current project
                self.project_data = new_project
                self._save_current_project_id(project_id)
                
                self.logger.info(f"Project created successfully: {name} (ID: {project_id})")
                return project_id
            else:
                self.logger.error("Failed to save new project to database")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating project in DB: {e}")
            return None
    
    def create_project(self, name: str, author: str = "", description: str = "") -> bool:
        """Tạo project mới VÀ set làm current project"""
        try:
            # Tạo project trong database (đã tự động set current)
            project_id = self._create_project_in_db(name, author, description)
            
            if project_id:
                self.logger.info(f"Created and set current project: {name} (ID: {project_id})")
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            return False
    
    def switch_project(self, project_id: str) -> bool:
        """Chuyển sang project khác"""
        try:
            projects_db = self._load_projects_db()
            
            if project_id not in projects_db:
                self.logger.error(f"Project {project_id} not found in database")
                return False
            
            # Set current project
            self.project_data = projects_db[project_id]
            self._save_current_project_id(project_id)
            
            self.logger.info(f"Switched to project: {self.project_data.get('name')} (ID: {project_id})")
            return True
        except Exception as e:
            self.logger.error(f"Error switching project: {e}")
            return False