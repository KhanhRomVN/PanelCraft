import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor

@dataclass
class ColorSettings:
    """Color settings matching React theme structure"""
    primary: str = "#3686ff"
    background: str = "#ffffff"
    text_primary: str = "#0f172a"
    text_secondary: str = "#475569"
    border: str = "#e2e8f0"
    border_hover: str = "#cbd5e1"
    border_focus: str = "#cbd5e1"
    card_background: str = "#ffffff"
    input_background: str = "#ffffff"
    modal_background: str = "#ffffff"
    dropdown_background: str = "#ffffff"
    dropdown_item_hover: str = "#f8fafc"
    sidebar_background: str = "#f9fafb"
    sidebar_item_hover: str = "#f3f4f6"
    sidebar_item_focus: str = "#e5e7eb"
    button_bg: str = "#3686ff"
    button_bg_hover: str = "#1d4ed8"
    button_text: str = "#ffffff"
    button_border: str = "#2563eb"
    button_border_hover: str = "#1e40af"
    button_second_bg: str = "#d4d4d4"
    button_second_bg_hover: str = "#b6b6b6"
    bookmark_item_bg: str = "#f1f5f9"
    bookmark_item_text: str = "#0f172a"
    drawer_background: str = "#ffffff"
    clock_gradient_from: str = "#3686ff"
    clock_gradient_to: str = "#1d4ed8"
    card_shadow: Optional[str] = None
    modal_shadow: Optional[str] = None
    dropdown_shadow: Optional[str] = None

class ThemePalette:
    """Theme palette matching React structure"""
    
    def __init__(self, name: str, color_settings: ColorSettings, description: str = ""):
        self.name = name
        self.colors = color_settings
        self.description = description

class ThemePresets:
    """Preset themes matching React structure"""
    
    @staticmethod
    def get_light_themes() -> List[ThemePalette]:
        return [
            ThemePalette(
                "Default Light",
                ColorSettings(
                    primary="#3686ff",
                    background="#ffffff",
                    text_primary="#0f172a",
                    text_secondary="#475569",
                    border="#e2e8f0",
                    border_hover="#cbd5e1",
                    border_focus="#cbd5e1",
                    card_background="#ffffff",
                    input_background="#ffffff",
                    modal_background="#ffffff",
                    dropdown_background="#ffffff",
                    dropdown_item_hover="#f8fafc",
                    sidebar_background="#f9fafb",
                    sidebar_item_hover="#f3f4f6",
                    sidebar_item_focus="#e5e7eb",
                    button_bg="#3686ff",
                    button_bg_hover="#1d4ed8",
                    button_text="#ffffff",
                    button_border="#2563eb",
                    button_border_hover="#1e40af",
                    button_second_bg="#d4d4d4",
                    button_second_bg_hover="#b6b6b6",
                    bookmark_item_bg="#f1f5f9",
                    bookmark_item_text="#0f172a",
                    drawer_background="#ffffff",
                    clock_gradient_from="#3686ff",
                    clock_gradient_to="#1d4ed8"
                ),
                "Clean light theme with blue accents"
            ),
            ThemePalette(
                "Indigo Light",
                ColorSettings(
                    primary="#4f46e5",
                    background="#eef2ff",
                    text_primary="#3730a3",
                    text_secondary="#4338ca",
                    border="#c7d2fe",
                    border_hover="#a5b4fc",
                    border_focus="#a5b4fc",
                    card_background="#ffffff",
                    input_background="#ffffff",
                    modal_background="#ffffff",
                    dropdown_background="#ffffff",
                    dropdown_item_hover="#e0e7ff",
                    sidebar_background="#e0e7ff",
                    sidebar_item_hover="#c7d2fe",
                    sidebar_item_focus="#a5b4fc",
                    button_bg="#4f46e5",
                    button_bg_hover="#4338ca",
                    button_text="#ffffff",
                    button_border="#4338ca",
                    button_border_hover="#3730a3",
                    button_second_bg="#e0e7ff",
                    button_second_bg_hover="#c7d2fe",
                    bookmark_item_bg="#e0e7ff",
                    bookmark_item_text="#3730a3",
                    drawer_background="#ffffff",
                    clock_gradient_from="#4f46e5",
                    clock_gradient_to="#4338ca"
                ),
                "Indigo themed light variant"
            )
        ]
    
    @staticmethod
    def get_dark_themes() -> List[ThemePalette]:
        return [
            ThemePalette(
                "Default Dark",
                ColorSettings(
                    primary="#3686ff",
                    background="#0a0a0a",
                    text_primary="#ececec",
                    text_secondary="#a8a8a8",
                    border="#353535",
                    border_hover="#418dfe",
                    border_focus="#418dfe",
                    card_background="#242424",
                    input_background="#1e1e1e",
                    modal_background="#1e1e1e",
                    dropdown_background="#1e1e1e",
                    dropdown_item_hover="#2d2d2d",
                    sidebar_background="#131313",
                    sidebar_item_hover="#1e1e1e",
                    sidebar_item_focus="#333333",
                    button_bg="#3686ff",
                    button_bg_hover="#418dfe",
                    button_text="#ffffff",
                    button_border="#418dfe",
                    button_border_hover="#5aa3ff",
                    button_second_bg="#1e1e1e",
                    button_second_bg_hover="#343434",
                    bookmark_item_bg="#1e293b",
                    bookmark_item_text="#e2e8f0",
                    drawer_background="#1e1e1e",
                    clock_gradient_from="#3686ff",
                    clock_gradient_to="#418dfe"
                ),
                "Modern dark theme"
            ),
            ThemePalette(
                "Midnight Dark",
                ColorSettings(
                    primary="#6366f1",
                    background="#020617",
                    text_primary="#e2e8f0",
                    text_secondary="#94a3b8",
                    border="#1e293b",
                    border_hover="#6366f1",
                    border_focus="#6366f1",
                    card_background="#0f172a",
                    input_background="#1e293b",
                    modal_background="#0f172a",
                    dropdown_background="#1e293b",
                    dropdown_item_hover="#334155",
                    sidebar_background="#0b0e2a",
                    sidebar_item_hover="#0f172a",
                    sidebar_item_focus="#1e293b",
                    button_bg="#6366f1",
                    button_bg_hover="#4f46e5",
                    button_text="#ffffff",
                    button_border="#4f46e5",
                    button_border_hover="#4338ca",
                    button_second_bg="#1e293b",
                    button_second_bg_hover="#334155",
                    bookmark_item_bg="#1e293b",
                    bookmark_item_text="#e2e8f0",
                    drawer_background="#0f172a",
                    clock_gradient_from="#6366f1",
                    clock_gradient_to="#4f46e5"
                ),
                "Deep midnight blue theme"
            )
        ]

class ThemeManager(QObject):
    """Theme manager similar to React ThemeProvider"""
    
    theme_changed = Signal()
    
    def __init__(self, storage_key: str = "panelcraft-theme"):
        super().__init__()
        self.storage_key = storage_key
        self._theme_mode = "light"  # light, dark, system
        self._current_palette: Optional[ThemePalette] = None
        self._custom_colors: Optional[ColorSettings] = None
        
        self.load_theme()
    
    @property
    def theme_mode(self) -> str:
        return self._theme_mode
    
    @theme_mode.setter
    def theme_mode(self, mode: str):
        if mode not in ["light", "dark", "system"]:
            raise ValueError("Theme mode must be 'light', 'dark', or 'system'")
        self._theme_mode = mode
        self.apply_theme()
        self.save_theme()
        self.theme_changed.emit()
    
    @property
    def effective_theme(self) -> str:
        """Resolve system theme to actual theme"""
        if self._theme_mode == "system":
            # In PySide6, we can detect system dark mode
            # This is a simplified approach - you might want more sophisticated detection
            return "dark"  # Default to dark for system for now
        return self._theme_mode
    
    @property
    def current_colors(self) -> ColorSettings:
        """Get current color settings"""
        if self._custom_colors:
            return self._custom_colors
        
        if self._current_palette:
            return self._current_palette.colors
        
        # Return default based on effective theme
        presets = ThemePresets.get_light_themes() if self.effective_theme == "light" else ThemePresets.get_dark_themes()
        return presets[0].colors
    
    def get_preset_themes(self) -> List[ThemePalette]:
        """Get available preset themes for current mode"""
        if self.effective_theme == "light":
            return ThemePresets.get_light_themes()
        else:
            return ThemePresets.get_dark_themes()
    
    def set_preset_theme(self, theme_name: str):
        """Set theme from preset"""
        presets = self.get_preset_themes()
        for preset in presets:
            if preset.name == theme_name:
                self._current_palette = preset
                self._custom_colors = None
                self.apply_theme()
                self.save_theme()
                self.theme_changed.emit()
                break
    
    def set_custom_colors(self, colors: ColorSettings):
        """Set custom color settings"""
        self._custom_colors = colors
        self._current_palette = None
        self.apply_theme()
        self.save_theme()
        self.theme_changed.emit()
    
    def apply_theme(self):
        """Apply current theme to application"""
        colors = self.current_colors
        
        # Generate QSS with CSS variables similar to React
        qss = self._generate_qss(colors)
        
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.setStyleSheet(qss)
    
    def _generate_qss(self, colors: ColorSettings) -> str:
        """Generate QSS stylesheet with direct color values"""
        
        qss = f"""
        /* Application-wide styles */
        QMainWindow {{
            background-color: {colors.background};
            color: {colors.text_primary};
        }}
        
        QWidget {{
            background-color: {colors.background};
            color: {colors.text_primary};
            font-family: "Segoe UI", "Arial", sans-serif;
        }}
        
        /* Header */
        Header QLabel {{
            color: {colors.text_primary};
            font-size: 18px;
            font-weight: bold;
        }}
        
        Header QPushButton {{
            background-color: {colors.button_second_bg};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            border-radius: 4px;
            padding: 8px 16px;
        }}
        
        Header QPushButton:hover {{
            background-color: {colors.button_second_bg_hover};
            border-color: {colors.border_hover};
        }}
        
        /* Sidebar */
        Sidebar {{
            background-color: {colors.sidebar_background};
            border-right: 1px solid {colors.border};
        }}
        
        Sidebar QPushButton {{
            background-color: transparent;
            color: {colors.text_primary};
            border: none;
            border-radius: 5px;
            padding: 10px;
            text-align: left;
        }}
        
        Sidebar QPushButton:checked {{
            background-color: {colors.primary};
            color: {colors.button_text};
        }}
        
        Sidebar QPushButton:hover {{
            background-color: {colors.sidebar_item_hover};
        }}
        
        /* Content areas */
        QTabWidget::pane {{
            border: 1px solid {colors.border};
            background-color: {colors.card_background};
        }}
        
        QTabBar::tab {{
            background-color: {colors.button_second_bg};
            color: {colors.text_primary};
            padding: 8px 16px;
            border: 1px solid {colors.border};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors.primary};
            color: {colors.button_text};
        }}
        
        QTabBar::tab:hover {{
            background-color: {colors.button_second_bg_hover};
        }}
        
        /* Buttons */
        QPushButton {{
            background-color: {colors.button_bg};
            color: {colors.button_text};
            border: 1px solid {colors.button_border};
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: {colors.button_bg_hover};
            border-color: {colors.button_border_hover};
        }}
        
        QPushButton[styleClass="secondary"] {{
            background-color: {colors.button_second_bg};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
        }}
        
        QPushButton[styleClass="secondary"]:hover {{
            background-color: {colors.button_second_bg_hover};
        }}
        
        /* Inputs */
        QLineEdit, QComboBox, QSpinBox {{
            background-color: {colors.input_background};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            border-radius: 4px;
            padding: 6px 8px;
        }}
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
            border-color: {colors.border_focus};
            outline: none;
        }}
        
        QComboBox::drop-down {{
            border: none;
            background-color: {colors.button_second_bg};
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {colors.text_primary};
            margin-right: 5px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors.dropdown_background};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            selection-background-color: {colors.dropdown_item_hover};
            selection-color: {colors.text_primary};
        }}
        
        /* Tables */
        QTableWidget {{
            background-color: {colors.card_background};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            border-radius: 4px;
            gridline-color: {colors.border};
        }}
        
        QHeaderView::section {{
            background-color: {colors.sidebar_background};
            color: {colors.text_primary};
            padding: 8px;
            border: none;
            font-weight: bold;
        }}
        
        /* Group boxes */
        QGroupBox {{
            background-color: {colors.card_background};
            color: {colors.text_primary};
            border: 1px solid {colors.border};
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }}
        
        /* Modals */
        QModal {{
            background-color: {colors.modal_background};
            color: {colors.text_primary};
        }}
        
        QMessageBox {{
            background-color: {colors.modal_background};
            color: {colors.text_primary};
        }}
        
        /* Status bar */
        QStatusBar {{
            background-color: {colors.sidebar_background};
            color: {colors.text_secondary};
            border-top: 1px solid {colors.border};
        }}
        
        /* Labels in content */
        QLabel {{
            background-color: transparent;
            color: {colors.text_primary};
        }}
        """
        
        return qss
    
    def save_theme(self):
        """Save theme settings to storage"""
        try:
            settings = {
                "theme_mode": self._theme_mode,
                "current_palette": self._current_palette.name if self._current_palette else None,
                "custom_colors": asdict(self._custom_colors) if self._custom_colors else None
            }
            
            # Using QSettings for storage (you can modify to use JSON file)
            from PySide6.QtCore import QSettings
            storage = QSettings("PanelCraft", "Theme")
            storage.setValue(self.storage_key, json.dumps(settings))
            
        except Exception as e:
            print(f"Error saving theme: {e}")
    
    def load_theme(self):
        """Load theme settings from storage"""
        try:
            from PySide6.QtCore import QSettings
            storage = QSettings("PanelCraft", "Theme")
            data = storage.value(self.storage_key)
            
            if data:
                settings = json.loads(data)
                self._theme_mode = settings.get("theme_mode", "light")
                
                palette_name = settings.get("current_palette")
                if palette_name:
                    presets = self.get_preset_themes()
                    for preset in presets:
                        if preset.name == palette_name:
                            self._current_palette = preset
                            break
                
                custom_colors = settings.get("custom_colors")
                if custom_colors:
                    self._custom_colors = ColorSettings(**custom_colors)
                
                self.apply_theme()
                
        except Exception as e:
            print(f"Error loading theme: {e}")
            # Apply default theme
            self.apply_theme()