STYLESHEET_ACADEMIC = """
QMainWindow, QWidget {
    background-color: #2b2d30;
    color: #cfd8dc;
    font-family: 'Roboto', 'Segoe UI';
}
QGroupBox {
    border: 1px solid #4e5154;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
}
QGroupBox::title { color: #589df6; left: 10px; }

QPushButton {
    background-color: #365880;
    color: #ffffff;
    border-radius: 4px;
    padding: 5px;
}
QPushButton:hover { background-color: #436996; }

QComboBox, QSpinBox, QLineEdit {
    background-color: #1e1f22;
    border: 1px solid #4e5154;
    border-radius: 3px;
    color: #bcbec4;
}
QTabWidget::pane { border: 1px solid #4e5154; top: -1px; }
QTabBar::tab {
    background: #2b2d30;
    padding: 8px 20px;
    border: 1px solid #4e5154;
    border-bottom: none;
}
QTabBar::tab:selected { background: #1e1f22; border-top: 2px solid #589df6; }

QTextEdit { 
    background-color: #1e1f22; 
    border: none; 
    color: #aabbcc; 
    font-family: 'JetBrains Mono', monospace; 
}
QListWidget { background-color: #2b2d30; border: 1px solid #4e5154; outline: none; }
QListWidget::item:selected { background-color: #2d4f73; }
"""

STYLESHEET_CYBERPUNK = """
QMainWindow, QWidget {
    background-color: #0b0e14;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
QGroupBox {
    border: 1px solid #00f2ff;
    border-radius: 4px;
    margin-top: 20px;
    background-color: #161b22;
}
QGroupBox::title { color: #00f2ff; subcontrol-origin: margin; left: 10px; }

QPushButton {
    background-color: #161b22;
    color: #00f2ff;
    border: 1px solid #00f2ff;
    border-radius: 2px;
    font-weight: bold;
    text-transform: uppercase;
}
QPushButton:hover { background-color: #00f2ff; color: #0b0e14; }

QComboBox, QSpinBox, QLineEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    color: #00f2ff;
    padding: 4px;
}
QTabWidget::pane { border: 1px solid #30363d; }
QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    padding: 10px;
    border: 1px solid #30363d;
}
QTabBar::tab:selected { color: #00f2ff; border-bottom: 2px solid #00f2ff; background: #0d1117; }

QTextEdit { background-color: #010409; color: #39ff14; border: 1px solid #30363d; font-family: 'Courier New'; }
QScrollBar:vertical { background: #0d1117; width: 8px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; }
"""

STYLESHEET_CLINICAL = """
QMainWindow, QWidget {
    background-color: #f5f7f9;
    color: #333333;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d1d9e0;
    border-radius: 8px;
    margin-top: 1.5em;
    font-weight: bold;
    color: #005a9e;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #0078d4;
    color: white;
    border-radius: 4px;
    padding: 6px 12px;
    border: none;
}
QPushButton:hover { background-color: #106ebe; }
QPushButton:pressed { background-color: #005a9e; }

QComboBox, QSpinBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #ccd0d4;
    border-radius: 4px;
    padding: 4px;
}
QTabWidget::pane { border: 1px solid #d1d9e0; background: white; border-radius: 4px; }
QTabBar::tab {
    background: #e1e4e8;
    padding: 8px 15px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background: white; border-bottom: 2px solid #0078d4; font-weight: bold; }

QTextEdit { background-color: #ffffff; border: 1px solid #d1d9e0; font-family: 'Consolas'; font-size: 11px; }
QScrollBar:vertical { border: none; background: #f0f0f0; width: 10px; }
QScrollBar::handle:vertical { background: #cdcdcd; border-radius: 5px; }
"""