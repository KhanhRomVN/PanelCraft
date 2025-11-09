def get_toggle_button_style() -> str:
    """Style cho toggle buttons"""
    return """
        QPushButton {
            background-color: var(--card-background);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: var(--sidebar-background);
            border-color: var(--primary);
        }
        QPushButton:checked {
            background-color: var(--primary);
            color: white;
            border-color: var(--primary);
        }
    """


def create_nav_button_style() -> str:
    """Style cho navigation buttons"""
    return """
        QPushButton {
            background-color: var(--card-background);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: var(--sidebar-background);
            border-color: var(--primary);
        }
        QPushButton:pressed {
            background-color: var(--primary);
        }
    """