"""
File storage abstraction (placeholder).
Future: local FS, S3, etc.
"""
class Storage:
    def save(self, src: str, dest: str):
        raise NotImplementedError
