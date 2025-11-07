from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QSpinBox, QPushButton, QGroupBox,
                              QFormLayout, QColorDialog, QGridLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QPainter

from core.theme import ThemeManager, ThemePresets, ColorSettings

class ColorButton(QPushButton):
    """Color picker button"""
    
    color_changed = Signal(str)
    
    def __init__(self, color: str = "#000000"):
        super().__init__()
        self._color = color
        self.setFixedSize(30, 30)
        self.update_color_display()
        self.clicked.connect(self.pick_color)
    
    def update_color_display(self):
        """Update button appearance with current color"""
        pixmap = QPixmap(30, 30)
        pixmap.fill(QColor(self._color))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor("#ccc"))
        painter.drawRect(0, 0, 29, 29)
        painter.end()
        
        self.setIcon(pixmap)
    
    def pick_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(QColor(self._color), self, "Choose Color")
        if color.isValid():
            self._color = color.name()
            self.update_color_display()
            self.color_changed.emit(self._color)
    
    @property
    def color(self) -> str:
        return self._color
    
    @color.setter
    def color(self, value: str):
        self._color = value
        self.update_color_display()

class ThemeSettingsWidget(QWidget):
    """Theme settings section"""
    
    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        self.theme_manager = theme_manager
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Theme mode selection
        mode_group = QGroupBox("Theme Mode")
        mode_layout = QFormLayout(mode_group)
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Light", "Dark", "System"])
        mode_layout.addRow("Mode:", self.mode_combo)
        
        layout.addWidget(mode_group)
        
        # Preset themes
        preset_group = QGroupBox("Preset Themes")
        preset_layout = QGridLayout(preset_group)
        
        self.preset_buttons = []
        presets = self.theme_manager.get_preset_themes()
        
        for i, preset in enumerate(presets):
            btn = QPushButton(preset.name)
            btn.setCheckable(True)
            btn.setProperty("theme_name", preset.name)
            btn.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding: 10px;
                    border: 2px solid {preset.colors.border};
                    border-radius: 4px;
                    background-color: {preset.colors.card_background};
                    color: {preset.colors.text_primary};
                }}
                QPushButton:checked {{
                    border-color: {preset.colors.primary};
                    background-color: {preset.colors.sidebar_item_hover};
                }}
                QPushButton:hover {{
                    background-color: {preset.colors.sidebar_item_hover};
                }}
            """)
            preset_layout.addWidget(btn, i // 2, i % 2)
            self.preset_buttons.append(btn)
        
        layout.addWidget(preset_group)
        
        # Custom colors
        self.custom_group = QGroupBox("Custom Colors")
        custom_layout = QGridLayout(self.custom_group)
        
        self.color_buttons = {}
        colors = self.theme_manager.current_colors
        color_fields = [
            ("Primary", "primary", colors.primary),
            ("Background", "background", colors.background),
            ("Text Primary", "text_primary", colors.text_primary),
            ("Card Background", "card_background", colors.card_background),
            ("Border", "border", colors.border),
            ("Button BG", "button_bg", colors.button_bg),
        ]
        
        for i, (label, key, color) in enumerate(color_fields):
            lbl = QLabel(label)
            btn = ColorButton(color)
            self.color_buttons[key] = btn
            
            custom_layout.addWidget(lbl, i // 2, (i % 2) * 2)
            custom_layout.addWidget(btn, i // 2, (i % 2) * 2 + 1)
        
        self.apply_custom_btn = QPushButton("Apply Custom Colors")
        self.apply_custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        
        custom_layout.addWidget(self.apply_custom_btn, (len(color_fields) + 1) // 2, 0, 1, 4)
        
        layout.addWidget(self.custom_group)
        layout.addStretch()
        
    def setup_connections(self):
        """Setup signal connections"""
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        
        for btn in self.preset_buttons:
            btn.clicked.connect(self.on_preset_selected)
        
        self.apply_custom_btn.clicked.connect(self.apply_custom_colors)
        
    def on_mode_changed(self, mode: str):
        """Handle theme mode change"""
        mode_map = {"Light": "light", "Dark": "dark", "System": "system"}
        self.theme_manager.theme_mode = mode_map.get(mode, "light")
        self.refresh_presets()
        
    def on_preset_selected(self):
        """Handle preset theme selection"""
        sender = self.sender()
        if sender.isChecked():
            # Uncheck other buttons
            for btn in self.preset_buttons:
                if btn != sender:
                    btn.setChecked(False)
            
            theme_name = sender.property("theme_name")
            self.theme_manager.set_preset_theme(theme_name)
    
    def apply_custom_colors(self):
        """Apply custom color settings"""
        colors_dict = {}
        for key, btn in self.color_buttons.items():
            colors_dict[key] = btn.color
        
        colors = ColorSettings(**colors_dict)
        self.theme_manager.set_custom_colors(colors)
    
    def refresh_presets(self):
        """Refresh preset themes based on current mode"""
        presets = self.theme_manager.get_preset_themes()
        
        for i, (btn, preset) in enumerate(zip(self.preset_buttons, presets)):
            btn.setText(preset.name)
            btn.setProperty("theme_name", preset.name)
            btn.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding: 10px;
                    border: 2px solid {preset.colors.border};
                    border-radius: 4px;
                    background-color: {preset.colors.card_background};
                    color: {preset.colors.text_primary};
                }}
                QPushButton:checked {{
                    border-color: {preset.colors.primary};
                    background-color: {preset.colors.sidebar_item_hover};
                }}
                QPushButton:hover {{
                    background-color: {preset.colors.sidebar_item_hover};
                }}
            """)
            
            # Check if this is the current theme
            if (self.theme_manager._current_palette and 
                self.theme_manager._current_palette.name == preset.name):
                btn.setChecked(True)
            else:
                btn.setChecked(False)

class SettingsWidget(QWidget):
    """Main settings widget containing all settings sections"""
    
    def __init__(self, config, theme_manager):
        super().__init__()
        self.config = config
        self.theme_manager = theme_manager
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Settings")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        layout.addWidget(title_label)
        
        # Theme settings section
        self.theme_settings = ThemeSettingsWidget(self.theme_manager)
        layout.addWidget(self.theme_settings)
        
        layout.addStretch()