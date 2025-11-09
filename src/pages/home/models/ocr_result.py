from dataclasses import dataclass
from typing import Optional


@dataclass
class OCRResult:
    """Data model cho OCR result của một segment"""
    
    segment_id: int
    original_text: str
    translated_text: str = ""
    character_id: Optional[int] = None
    character_name: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'segment_id': self.segment_id,
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'character_id': self.character_id,
            'character_name': self.character_name,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OCRResult':
        """Create from dictionary"""
        return cls(
            segment_id=data['segment_id'],
            original_text=data['original_text'],
            translated_text=data.get('translated_text', ''),
            character_id=data.get('character_id'),
            character_name=data.get('character_name'),
            confidence=data.get('confidence', 0.0)
        )
    
    def is_valid(self) -> bool:
        """Check if OCR result is valid"""
        return bool(self.original_text.strip())
    
    def get_display_text(self, max_length: int = 50) -> str:
        """Get truncated text for display"""
        text = self.original_text
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."