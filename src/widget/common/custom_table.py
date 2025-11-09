from PySide6.QtWidgets import (QTableWidget, QTableWidgetItem, QVBoxLayout, 
                              QHBoxLayout, QWidget, QLabel, QPushButton, 
                              QHeaderView, QAbstractItemView, QProgressBar, QSizePolicy)
from PySide6.QtCore import Qt, Signal, Property, QTimer
from PySide6.QtGui import QFont, QColor, QBrush
from typing import List, Dict, Any, Optional

class CustomTable(QWidget):
    """Custom table component with pagination and loading states"""
    
    # Signals
    pageChanged = Signal(int)
    rowClicked = Signal(int, dict)
    rowDoubleClicked = Signal(int, dict)
    
    def __init__(self,
                 headers: List[str] = None,
                 data: List[Dict[str, Any]] = None,
                 page_size: int = 10,
                 show_pagination: bool = True,
                 loading: bool = False,
                 parent=None):
        super().__init__(parent)
        
        self._headers = headers or []
        self._data = data or []
        self._page_size = page_size
        self._show_pagination = show_pagination
        self._loading = loading
        self._current_page = 1
        self._total_pages = 1
        
        self.setup_ui()
        self.apply_styles()
        self.setup_connections()
        self.update_table()
        
    def setup_ui(self):
        """Setup table layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self._headers))
        self.table_widget.setHorizontalHeaderLabels(self._headers)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setAlternatingRowColors(True)
        
        # ẨN vertical header (cột row numbers bên trái)
        self.table_widget.verticalHeader().setVisible(False)
        
        # Configure header
        header = self.table_widget.horizontalHeader()
        
        # Set resize modes cho các cột
        if len(self._headers) > 0:
            # Cột đầu tiên (STT) - Fit content
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        
        if len(self._headers) > 1:
            # Cột thứ 2 (Character) - Fit content
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        # Các cột còn lại - Interactive
        for i in range(2, len(self._headers)):
            header.setSectionResizeMode(i, QHeaderView.Interactive)
        
        # Cột cuối - Stretch
        if len(self._headers) > 2:
            header.setSectionResizeMode(len(self._headers) - 1, QHeaderView.Stretch)
        
        layout.addWidget(self.table_widget)
        
        # Pagination container
        self.pagination_container = QWidget()
        pagination_layout = QHBoxLayout(self.pagination_container)
        pagination_layout.setContentsMargins(16, 12, 16, 12)
        pagination_layout.setSpacing(12)
        
        # Page info
        self.page_info_label = QLabel()
        
        # Pagination buttons
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        pagination_layout.addWidget(self.page_info_label)
        pagination_layout.addWidget(spacer)
        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addWidget(self.next_button)
        
        layout.addWidget(self.pagination_container)
        
        # Set initial visibility
        self.pagination_container.setVisible(self._show_pagination)
    
    def apply_styles(self):
        """Apply table styles"""
        table_style = """
            QTableWidget {
                background-color: var(--card-background);
                border: 1px solid var(--border);
                border-radius: 8px;
                gridline-color: var(--border);
                outline: none;
            }
            QTableWidget::item {
                padding: 10px 12px;
                border: none;
                color: var(--text-primary);
                height: 32px;
            }
            QTableWidget::item:selected {
                background-color: var(--primary);
                color: white;
            }
            QTableWidget::item:hover {
                background-color: var(--sidebar-item-hover);
            }
            QHeaderView::section {
                background-color: var(--sidebar-background);
                padding: 10px 12px;
                border: none;
                border-bottom: 2px solid var(--border);
                font-weight: bold;
                color: var(--text-primary);
            }
            QHeaderView::section:hover {
                background-color: var(--sidebar-item-hover);
            }
        """
        
        self.table_widget.setStyleSheet(table_style)
        
        # Pagination button styles
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: 500;
                min-height: 36px;
            }
        """
        
        enabled_style = """
            QPushButton {
                background-color: var(--button-bg);
                color: var(--button-text);
                border: 1px solid var(--button-border);
            }
            QPushButton:hover {
                background-color: var(--button-bg-hover);
            }
        """
        
        disabled_style = """
            QPushButton {
                background-color: var(--button-second-bg);
                color: var(--text-secondary);
                border: 1px solid var(--border);
                opacity: 0.5;
            }
        """
        
        self.prev_button.setStyleSheet(button_style + disabled_style)
        self.next_button.setStyleSheet(button_style + disabled_style)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.prev_button.clicked.connect(self.previous_page)
        self.next_button.clicked.connect(self.next_page)
        self.table_widget.itemClicked.connect(self.on_item_clicked)
        self.table_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
    
    def update_table(self):
        """Update table with current data"""
        if self._loading:
            self.show_loading()
            return
            
        # Calculate pagination
        start_idx = (self._current_page - 1) * self._page_size
        end_idx = start_idx + self._page_size
        page_data = self._data[start_idx:end_idx]
        
        # Set CHÍNH XÁC số dòng (không thêm dòng trống)
        self.table_widget.setRowCount(len(page_data))
        
        # Clear tất cả items cũ trước khi populate
        self.table_widget.clearContents()
        
        # Populate table
        for row, item in enumerate(page_data):
            for col, header in enumerate(self._headers):
                value = str(item.get(header, ""))
                table_item = QTableWidgetItem(value)
                
                # Center align STT column
                if col == 0:
                    table_item.setTextAlignment(Qt.AlignCenter)
                
                # Elide text cho các cột text dài (không phải STT, Character)
                if col >= 2:
                    # Giới hạn độ dài hiển thị
                    if len(value) > 50:
                        display_value = value[:47] + "..."
                        table_item.setText(display_value)
                        table_item.setToolTip(value)  # Hiển thị full text khi hover
                
                self.table_widget.setItem(row, col, table_item)
        
        # Update pagination
        self.update_pagination()
    
    def update_pagination(self):
        """Update pagination controls"""
        total_items = len(self._data)
        self._total_pages = max(1, (total_items + self._page_size - 1) // self._page_size)
        
        # Update page info
        start_idx = (self._current_page - 1) * self._page_size + 1
        end_idx = min(self._current_page * self._page_size, total_items)
        
        if total_items > 0:
            self.page_info_label.setText(
                f"Showing {start_idx}-{end_idx} of {total_items} items"
            )
        else:
            self.page_info_label.setText("No items to display")
        
        # Update button states
        self.prev_button.setEnabled(self._current_page > 1)
        self.next_button.setEnabled(self._current_page < self._total_pages)
        
        # Update button styles
        prev_style = """
            QPushButton {
                background-color: %s;
                color: %s;
                border: 1px solid %s;
            }
        """ % (
            "var(--button-bg)" if self._current_page > 1 else "var(--button-second-bg)",
            "var(--button-text)" if self._current_page > 1 else "var(--text-secondary)",
            "var(--button-border)" if self._current_page > 1 else "var(--border)"
        )
        
        next_style = """
            QPushButton {
                background-color: %s;
                color: %s;
                border: 1px solid %s;
            }
        """ % (
            "var(--button-bg)" if self._current_page < self._total_pages else "var(--button-second-bg)",
            "var(--button-text)" if self._current_page < self._total_pages else "var(--text-secondary)",
            "var(--button-border)" if self._current_page < self._total_pages else "var(--border)"
        )
        
        self.prev_button.setStyleSheet(prev_style)
        self.next_button.setStyleSheet(next_style)
    
    def show_loading(self):
        """Show loading state"""
        self.table_widget.setRowCount(5)  # Show skeleton rows
        self.table_widget.setColumnCount(len(self._headers))
        
        for row in range(5):
            for col in range(len(self._headers)):
                item = QTableWidgetItem("Loading...")
                item.setForeground(QBrush(QColor(200, 200, 200)))
                self.table_widget.setItem(row, col, item)
    
    def previous_page(self):
        """Go to previous page"""
        if self._current_page > 1:
            self._current_page -= 1
            self.update_table()
            self.pageChanged.emit(self._current_page)
    
    def next_page(self):
        """Go to next page"""
        if self._current_page < self._total_pages:
            self._current_page += 1
            self.update_table()
            self.pageChanged.emit(self._current_page)
    
    def on_item_clicked(self, item):
        """Handle item click"""
        row = item.row()
        page_data = self.get_current_page_data()
        if row < len(page_data):
            self.rowClicked.emit(row, page_data[row])
    
    def on_item_double_clicked(self, item):
        """Handle item double click"""
        row = item.row()
        page_data = self.get_current_page_data()
        if row < len(page_data):
            self.rowDoubleClicked.emit(row, page_data[row])
    
    def get_current_page_data(self) -> List[Dict[str, Any]]:
        """Get data for current page"""
        start_idx = (self._current_page - 1) * self._page_size
        end_idx = start_idx + self._page_size
        return self._data[start_idx:end_idx]
    
    # Public methods
    def setData(self, data: List[Dict[str, Any]]):
        """Set table data"""
        self._data = data
        self._current_page = 1
        self.update_table()
    
    def setHeaders(self, headers: List[str]):
        """Set table headers"""
        self._headers = headers
        self.table_widget.setColumnCount(len(headers))
        self.table_widget.setHorizontalHeaderLabels(headers)
        self.update_table()
    
    def setLoading(self, loading: bool):
        """Set loading state"""
        self._loading = loading
        if loading:
            self.show_loading()
        else:
            self.update_table()
    
    def clear(self):
        """Clear table"""
        self._data = []
        self._current_page = 1
        self.update_table()