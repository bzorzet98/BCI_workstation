import sys
import os
from src.interface.bciworkstation import BCIWorkstationGUI
from PyQt6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIWorkstationGUI(theme_name="CLINICAL")
    window.show()
    sys.exit(app.exec())
        