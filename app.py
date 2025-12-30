import sys
import os
from src.interface.bciworkstation import BCIWorkstationGUI
from PyQt6.QtWidgets import QApplication
from src.interface.themes import STYLESHEET_ACADEMIC, STYLESHEET_CYBERPUNK, STYLESHEET_CLINICAL
import matplotlib.pyplot as plt

theme_kwargs = {
    "STYLESHEET_CLINICAL":{
        "pyplot_style":{
            "figure.facecolor": "#f5f7f9",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d1d9e0",
            "axes.labelcolor": "#333333",
            "axes.prop_cycle": plt.cycler(color=['#0078d4', '#d13438', '#107c10', '#5c2d91', '#008272']),
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#e1e4e8",
            "text.color": "#333333"
        },
        "logo_path": os.path.join("src", "interface", "logo", "logo_system.png")
    },
    "STYLESHEET_CYBERPUNK":{
        "pyplot_style":{
            "figure.facecolor": "#0b0e14",
            "axes.facecolor": "#0d1117",
            "axes.edgecolor": "#00f2ff",
            "axes.labelcolor": "#00f2ff",
            "axes.prop_cycle": plt.cycler(color=['#00f2ff', '#39ff14', '#ff00ff', '#ffff00', '#ff3131']),
            "xtick.color": "#00f2ff",
            "ytick.color": "#00f2ff",
            "grid.color": "#161b22",
            "text.color": "#e0e0e0"
        },
        "logo_path": os.path.join("src", "interface", "logo", "logo_system2.png")
    },
    "STYLESHEET_ACADEMIC":{
        "pyplot_style":{
            "figure.facecolor": "#2b2d30",
            "axes.facecolor": "#1e1f22",
            "axes.edgecolor": "#4e5154",
            "axes.labelcolor": "#cfd8dc",
            "axes.prop_cycle": plt.cycler(color=['#589df6', '#81b1ff', '#4db6ac', '#ff8a65', '#a1887f']),
            "xtick.color": "#cfd8dc",
            "ytick.color": "#cfd8dc",
            "grid.color": "#2b2d30",
            "text.color": "#cfd8dc"
        },
        "logo_path": os.path.join("src", "interface", "logo", "logo_system2.png")
    }
}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 1. Set Style Strategy (Fusion works best cross-platform for dark mode)
    app.setStyle("Fusion")
    
    # 2. Apply your chosen stylesheet
    app.setStyleSheet(STYLESHEET_CLINICAL) 
    
    window = BCIWorkstationGUI(style_kwargs=theme_kwargs["STYLESHEET_CLINICAL"])
    window.show()
    sys.exit(app.exec())
    