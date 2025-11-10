from PySide6.QtWidgets import (QComboBox, QVBoxLayout, QHBoxLayout, QLabel, 
                              QWidget, QListWidget, QLineEdit, QListWidgetItem,
                              QAbstractItemView, QPushButton, QScrollArea)
from PySide6.QtCore import Qt, Signal, Property, QTimer, QSize
from PySide6.QtGui import QIcon, QFont
from typing import List, Dict, Optional, Tuple

class CustomCombobox(QWidget):
    """Custom combobox with search, multi-select, and creatable options"""
    
    # Signals
    valueChanged = Signal((str,), (list,))
    currentTextChanged = Signal(str)
    
    def __init__(self,
                 label: str = "",
                 options: List[Dict[str, str]] = None,
                 placeholder: str = "Select an option...",
                 searchable: bool = False,
                 multiple: bool = False,
                 creatable: bool = False,
                 size: str = "md",
                 required: bool = False,
                 parent=None):
        super().__init__(parent)
        
        self._label = label
        self._options = options or []
        self._placeholder = placeholder
        self._searchable = searchable
        self._multiple = multiple
        self._creatable = creatable
        self._size = size
        self._required = required
        self._selected_values = []
        self._selected_items = []
        
        self.is_dropdown_open = False
        self.filtered_options = self._options.copy()
        
        self.setup_ui()
        self.apply_styles()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup combobox layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Label
        if self._label:
            self.label_widget = QLabel(self._label)
            if self._required:
                self.label_widget.setText(f"{self._label} *")
            layout.addWidget(self.label_widget)
        
        # Main combobox container
        self.combobox_container = QWidget()
        combobox_layout = QHBoxLayout(self.combobox_container)
        combobox_layout.setContentsMargins(0, 0, 0, 0)
        
        # Combobox
        self.combobox = QComboBox()
        self.combobox.setEditable(self._searchable)
        
        # Set placeholder if searchable
        if self._searchable and self.combobox.lineEdit():
            self.combobox.lineEdit().setPlaceholderText(self._placeholder)
        
        # Populate options
        self.populate_options()
        
        combobox_layout.addWidget(self.combobox)
        
        # Multi-select badges container (hidden by default)
        self.badges_container = QWidget()
        self.badges_layout = QHBoxLayout(self.badges_container)
        self.badges_layout.setContentsMargins(0, 0, 0, 0)
        self.badges_layout.setSpacing(4)
        self.badges_container.hide()
        
        layout.addWidget(self.combobox_container)
        layout.addWidget(self.badges_container)
    
    def populate_options(self):
        """Populate combobox with options"""
        self.combobox.clear()
        
        # Add placeholder as first item if not searchable
        if not self._searchable:
            self.combobox.addItem(self._placeholder, "")
        
        # Add options
        for option in self._options:
            self.combobox.addItem(option.get("label", ""), option.get("value", ""))
    
    def apply_styles(self):
        """Apply styles based on size"""
        size_styles = {
            "sm": {
                "padding": "8px 12px",
                "font_size": "14px",
                "height": "32px"
            },
            "md": {
                "padding": "12px 16px",
                "font_size": "16px",
                "height": "40px"
            },
            "lg": {
                "padding": "16px 20px",
                "font_size": "18px",
                "height": "48px"
            }
        }
        
        size_config = size_styles.get(self._size, size_styles["md"])
        
        style = f"""
            QComboBox {{
                padding: {size_config['padding']};
                font-size: {size_config['font_size']};
                min-height: {size_config['height']};
                background-color: var(--input-background);
                border: 1px solid var(--border);
                border-radius: 6px;
                color: var(--text-primary);
            }}
            QComboBox:hover {{
                border-color: var(--primary);
            }}
            QComboBox:focus {{
                border-color: var(--primary);
                outline: none;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid var(--text-primary);
            }}
            QComboBox QAbstractItemView {{
                background-color: var(--dropdown-background);
                border: 1px solid var(--border);
                border-radius: 6px;
                selection-background-color: var(--dropdown-item-hover);
                color: var(--text-primary);
                outline: none;
            }}
        """
        
        try:
            self.combobox.setStyleSheet(style)
        except Exception as e:
            # Fallback nếu style không apply được
            self.combobox.setStyleSheet("QComboBox { background-color: white; }")
    
    def setup_connections(self):
        """Setup signal connections"""
        self.combobox.currentIndexChanged.connect(self.on_index_changed)
        
        # Connect search if searchable
        if self._searchable and self.combobox.lineEdit():
            self.combobox.lineEdit().textChanged.connect(self.on_search_changed)
    
    def on_index_changed(self, index):
        """Handle selection change"""
        if index < 0:
            return
            
        value = self.combobox.itemData(index)
        text = self.combobox.itemText(index)
        
        if self._multiple:
            if value and value not in self._selected_values:
                self._selected_values.append(value)
                self._selected_items.append(text)
                self.update_badges()
                
                # Clear selection but keep text if searchable
                if self._searchable:
                    self.combobox.setCurrentIndex(-1)
                    if self.combobox.lineEdit():
                        self.combobox.lineEdit().setText("")
                else:
                    self.combobox.setCurrentIndex(0)  # Reset to placeholder
                
            self.valueChanged[list].emit(self._selected_values)
        else:
            if value:  # Don't emit for placeholder
                self.valueChanged[str].emit(value)
                self.currentTextChanged.emit(text)
    
    def on_search_changed(self, text):
        """Handle search text changes"""
        if not text:
            self.filtered_options = self._options.copy()
            return
            
        # Filter options based on search text
        self.filtered_options = [
            opt for opt in self._options 
            if text.lower() in opt.get("label", "").lower() 
            or text.lower() in str(opt.get("value", "")).lower()
        ]
        
        # Update combobox with filtered options
        current_text = self.combobox.lineEdit().text() if self.combobox.lineEdit() else ""
        self.combobox.clear()
        
        for option in self.filtered_options:
            self.combobox.addItem(option.get("label", ""), option.get("value", ""))
        
        # Restore search text
        if self.combobox.lineEdit():
            self.combobox.lineEdit().setText(current_text)
    
    def update_badges(self):
        """Update multi-select badges"""
        # Clear existing badges
        for i in reversed(range(self.badges_layout.count())):
            widget = self.badges_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add badges for selected items
        for i, (value, text) in enumerate(zip(self._selected_values, self._selected_items)):
            badge = QPushButton(text)
            badge.setProperty("value", value)
            
            badge_style = """
                QPushButton {
                    background-color: var(--primary);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    padding: 4px 8px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: var(--button-bg-hover);
                }
            """
            badge.setStyleSheet(badge_style)
            badge.clicked.connect(lambda checked=False, v=value: self.remove_badge(v))
            
            self.badges_layout.addWidget(badge)
        
        # Show badges container if there are selections
        self.badges_container.setVisible(len(self._selected_values) > 0)
    
    def remove_badge(self, value):
        """Remove a badge from multi-select"""
        if value in self._selected_values:
            index = self._selected_values.index(value)
            self._selected_values.pop(index)
            self._selected_items.pop(index)
            self.update_badges()
            self.valueChanged[list].emit(self._selected_values)
    
    # Public methods
    def setOptions(self, options: List[Dict[str, str]]):
        """Set new options"""
        # Block ALL signals to prevent recursion
        was_blocked = self.signalsBlocked()
        combobox_was_blocked = self.combobox.signalsBlocked()
        
        self.blockSignals(True)
        self.combobox.blockSignals(True)
        
        try:
            self._options = options
            self.filtered_options = options.copy()
            self.populate_options()
        finally:
            # Restore original signal state
            self.blockSignals(was_blocked)
            self.combobox.blockSignals(combobox_was_blocked)
    
    def addOption(self, value: str, label: str):
        """Add a new option"""
        self._options.append({"value": value, "label": label})
        self.filtered_options.append({"value": value, "label": label})
        self.populate_options()
    
    def clear(self):
        """Clear selection"""
        self._selected_values.clear()
        self._selected_items.clear()
        self.update_badges()
        self.combobox.setCurrentIndex(0)
    
    def currentValue(self):
        """Get current value"""
        if self._multiple:
            return self._selected_values
        return self.combobox.currentData()
    
    def setCurrentValue(self, value):   
        """Set current value"""
        # ========== Recursion Guard ==========
        if hasattr(self, '_setting_value') and self._setting_value:
            return
        
        self._setting_value = True
        
        try:
            if self._multiple and isinstance(value, list):
                self._selected_values = value
                self._selected_items = [
                    opt.get("label", "") for opt in self._options 
                    if opt.get("value", "") in value
                ]
                self.update_badges()
            else:
                # Block ALL signals to prevent recursion
                was_blocked = self.signalsBlocked()
                combobox_was_blocked = self.combobox.signalsBlocked()
                
                self.blockSignals(True)
                self.combobox.blockSignals(True)
                
                try:
                    index = self.combobox.findData(value)
                    if index >= 0:
                        self.combobox.setCurrentIndex(index)
                finally:
                    # Restore original signal state
                    self.blockSignals(was_blocked)
                    self.combobox.blockSignals(combobox_was_blocked)
        finally:
            self._setting_value = False