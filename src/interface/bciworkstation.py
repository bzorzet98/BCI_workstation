import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QTabWidget, QGroupBox, QLabel, QFrame, QApplication)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import matplotlib.pyplot as plt

# --- CORE & WIDGET IMPORTS ---
from src.core.data_manager import DataManager
from src.core.session import SessionState
from src.interface.widgets.workspace_tab import WorkspaceTab
from src.interface.widgets.preprocessing_tabs import ProcessingTab
from src.interface.widgets.time_tab import TimeDomainTab
from src.interface.widgets.freq_tab import FreqDomainTab
from src.interface.widgets.eda_tab import EDATab
from src.interface.widgets.metadata_tab import DatasetMetadata
from src.interface.widgets.log_console import LogConsole
from src.interface.widgets.subject_info_tab import  SubjectInfoTab
from src.interface.themes import STYLESHEET_ACADEMIC, STYLESHEET_CYBERPUNK, STYLESHEET_CLINICAL


class BCIWorkstationGUI(QMainWindow):
    def __init__(self, theme_name="ACADEMIC"):
        super().__init__()
        
        # 1. DEFINE DYNAMIC THEME PARAMETERS
        # These will be passed to every Tab to style Matplotlib figures
        if theme_name == "CYBERPUNK":
            self.setStyleSheet(STYLESHEET_CYBERPUNK)
            self.style_kwargs = {
                "figure.facecolor": "#0b0e14",
                "axes.facecolor": "#0d1117",
                "axes.edgecolor": "#00f2ff",
                "axes.labelcolor": "#00f2ff",
                "axes.prop_cycle": plt.cycler(color=['#00f2ff', '#39ff14', '#ff00ff', '#ffff00', '#ff3131']),
                "xtick.color": "#00f2ff",
                "ytick.color": "#00f2ff",
                "grid.color": "#161b22",
                "text.color": "#e0e0e0"
            }
            self.logo_path = os.path.join("src", "interface", "logo", "logo_system2.png")
            
        elif theme_name == "CLINICAL":
            self.setStyleSheet(STYLESHEET_CLINICAL)
            self.style_kwargs = {
                "figure.facecolor": "#f5f7f9",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#d1d9e0",
                "axes.labelcolor": "#333333",
                "axes.prop_cycle": plt.cycler(color=['#0078d4', '#d13438', '#107c10', '#5c2d91', '#008272']),
                "xtick.color": "#333333",
                "ytick.color": "#333333",
                "grid.color": "#e1e4e8",
                "text.color": "#333333"
            }
            self.logo_path = os.path.join("src", "interface", "logo", "logo_system.png")
        else: # DEFAULT: ACADEMIC
            self.setStyleSheet(STYLESHEET_ACADEMIC)
            self.style_kwargs = {
                "figure.facecolor": "#2b2d30",
                "axes.facecolor": "#1e1f22",
                "axes.edgecolor": "#4e5154",
                "axes.labelcolor": "#cfd8dc",
                "axes.prop_cycle": plt.cycler(color=['#589df6', '#81b1ff', '#4db6ac', '#ff8a65', '#a1887f']),
                "xtick.color": "#cfd8dc",
                "ytick.color": "#cfd8dc",
                "grid.color": "#2b2d30",
                "text.color": "#cfd8dc"
            }
            self.logo_path = os.path.join("src", "interface", "logo", "logo_system2.png")

        self.setWindowTitle(f"BCI Workstation Professional [{theme_name}]")
        self.resize(1300, 950)

        # 2. CORE ENGINES
        self.data_manager = DataManager()
        self.session = SessionState()
        
        # --- UI CONSTRUCTION ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0) # Edge-to-edge look

        # Main Tab Container (occupies the top area)
        self.tabs = QTabWidget()
        self._setup_tabs()
        self.main_layout.addWidget(self.tabs, stretch=4)

        # Log Console (stays at the bottom)
        self._setup_log_area()

    def _setup_tabs(self):
        # We pass self.style_kwargs to EVERY tab so their Matplotlib plots match the theme
        self.tab_workspace = WorkspaceTab(self.session, self.data_manager, self.style_kwargs)
        self.tab_meta = DatasetMetadata(self.session, self.data_manager, self.style_kwargs)
        self.tab_proc = ProcessingTab(self.session, self.style_kwargs)
        self.tab_time = TimeDomainTab(self.session, self.style_kwargs)
        self.tab_freq = FreqDomainTab(self.session, self.style_kwargs)
        self.tab_eda = EDATab(self.session, self.style_kwargs)
        self.tab_subject_info = SubjectInfoTab(self.session, self.style_kwargs)
        
        self.tabs.addTab(self.tab_workspace, "Data Loader")
        self.tabs.addTab(self.tab_subject_info, "Subject Info")
        self.tabs.addTab(self.tab_meta, "Dataset Metadata")
        self.tabs.addTab(self.tab_proc, "Preprocessing")
        self.tabs.addTab(self.tab_time, "Time Domain")
        self.tabs.addTab(self.tab_freq, "Freq Domain")
        self.tabs.addTab(self.tab_eda, "EDA Analysis")

        # Connect events
        self.tabs.currentChanged.connect(self._on_tab_accessed)
        self.tab_workspace.dataLoaded.connect(self._on_data_loaded)
        self.tab_proc.stepApplied.connect(self.refresh_gui)
        
    def _on_tab_accessed(self, index):
        """ This is triggered when a tab is accessed. We use it to refresh plots."""
        label = self.tabs.tabText(index)
        if "Time Domain" in label:
            print("[GUI] Accessing Time Domain... refreshing plot.")
            self.tab_time.refresh()
        elif "Freq Domain" in label:
            print("[GUI] Accessing Frequency Domain...")
            self.tab_freq.refresh()
        elif "EDA Analysis" in label:
            print("[GUI] Accessing EDA Analysis...")
            self.tab_eda.refresh()
            
    def _setup_log_area(self):
        # We also style the group box title dynamically
        title_color = self.style_kwargs["text.color"]
        grp_log = QGroupBox("Execution Log")
        grp_log.setStyleSheet(f"QGroupBox::title {{ color: {title_color}; }}")
        
        v_log = QVBoxLayout(grp_log)
        self.console = LogConsole()
        v_log.addWidget(self.console)
        self.main_layout.addWidget(grp_log, stretch=1)
    # --- CONTROLLER LOGIC ---

    def _on_data_loaded(self):
        self.tab_subject_info.refresh() # Prueba 1: Descomenta solo esta y epoquea
        self.tab_meta.refresh()         # Prueba 2: Descomenta esta y epoquea
        self.tab_proc.refresh_options()
        self.refresh_gui()
        
    def refresh_gui(self):
        """Refresco diferido para evitar bloqueos de memoria."""
        # Usamos un timer de 50ms para asegurar que MNE liberó el puntero de memoria
        QTimer.singleShot(50, self._do_actual_refresh)

    def _do_actual_refresh(self):
        try:
            self.tab_subject_info.refresh()
            self.tab_meta.refresh()
            self.tab_proc.refresh_options() # Ahora con blockSignals no debería explotar
            print("[GUI] Global refresh completed successfully.")
        except Exception as e:
            print(f"[GUI] Error during refresh: {e}")