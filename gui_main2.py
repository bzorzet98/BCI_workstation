# gui_main.py

import sys
import ast
import json
import numpy as np
import mne
import pandas as pd
import pprint

# PyQt Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QSpinBox, QListWidget, QGroupBox, QLineEdit, 
                             QFormLayout, QMessageBox, QTabWidget, QScrollArea,
                             QGridLayout, QSizePolicy) # Added QSizePolicy for canvas

# --- CORE IMPORTS ---
from src.core.data_manager import DataManager
from src.core.session import SessionState
from src.core.registry import Registry
from src.interface.plotter import Plotter

# --- WIDGET IMPORTS ---
# Make sure these paths are correct based on your file structure
from src.interface.widgets.channel_tab import ChannelTab
from src.interface.widgets.info_tab import InfoTab
from src.interface.widgets.preprocessing_tabs import ProcessingTab
from src.interface.widgets.viz_tabs import VizTab # Import VizTab too

# --- MAIN WINDOW ---
class BCIWorkstationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Workstation Professional")
        self.resize(1200, 800)
        
        # --- MODEL ---
        self.data_manager = DataManager()
        self.session = SessionState()

        # --- LAYOUT ---
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # --- LEFT SIDEBAR ---
        sidebar = QVBoxLayout()
        w_side = QWidget()
        w_side.setFixedWidth(320)
        w_side.setLayout(sidebar)
        main_layout.addWidget(w_side)

        # 1. Data Selection Group
        grp_load = QGroupBox("Data Loading")
        form_load = QFormLayout()
        
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(self.data_manager.get_available_datasets())
        self.cb_dataset.currentTextChanged.connect(self.on_dataset_change)
        
        self.sb_subject = QSpinBox()
        self.sb_subject.setRange(1, 100) # Default, will be updated
        
        self.cb_context = QComboBox()
        
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.load_data)
        
        form_load.addRow("Dataset:", self.cb_dataset)
        form_load.addRow("Subject:", self.sb_subject)
        form_load.addRow("Context:", self.cb_context)
        form_load.addRow(btn_load)
        grp_load.setLayout(form_load)
        sidebar.addWidget(grp_load)

        # 2. Pipeline History Group
        grp_hist = QGroupBox("Pipeline History")
        v_hist = QVBoxLayout()
        self.list_history = QListWidget()
        v_hist.addWidget(self.list_history)
        
        btn_del = QPushButton("Delete Selected Step")
        btn_del.clicked.connect(self.delete_step_placeholder)
        v_hist.addWidget(btn_del)
        
        grp_hist.setLayout(v_hist)
        sidebar.addWidget(grp_hist)

        # --- RIGHT TABS AREA ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Instantiate Tab Widgets, PASSING SESSION AND DATAMANAGER
        self.tab_info = InfoTab(self.session, self.data_manager)
        self.tab_chans = ChannelTab(self.session)
        self.tab_proc = ProcessingTab(self.session)
        self.tab_viz = VizTab(self.session) # Pass session to VizTab for channel selection

        self.tabs.addTab(self.tab_info, "Info & Metadata")
        self.tabs.addTab(self.tab_chans, "Channels")
        self.tabs.addTab(self.tab_proc, "Preprocessing")
        self.tabs.addTab(self.tab_viz, "Visualization")

        # Initial population of UI elements
        if self.cb_dataset.count() > 0:
            self.on_dataset_change(self.cb_dataset.currentText())

    # --- LOGIC SLOTS ---

    def on_dataset_change(self, name):
        """Updates contexts and subject range when dataset selection changes."""
        if not name: return
        try:
            self.data_manager.load_dataset_class(name)
            meta = self.data_manager.get_metadata()
            
            self.cb_context.clear()
            if meta and 'contexts' in meta:
                self.cb_context.addItems(meta['contexts'])
            
            if meta and 'subjects' in meta:
                subs = meta['subjects']
                if subs:
                    self.sb_subject.setRange(min(subs), max(subs))
                    self.sb_subject.setValue(min(subs))
        except Exception as e:
            print(f"Error changing dataset: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load dataset metadata: {e}")

    def load_data(self):
        """Loads selected data and refreshes UI elements."""
        try:
            sub = self.sb_subject.value()
            ctx = self.cb_context.currentText()
            ds_name = self.cb_dataset.currentText()
            
            raw = self.data_manager.load_data(sub, ctx)
            
            if raw:
                # Fetch extra_info from dataset if available and store in session
                # (This relies on your dataset class having _get_single_subject_data that returns this)
                extra = self.data_manager.active_dataset_instance._get_single_subject_data(sub, [ctx]).get('extra_info', {})
                self.session.extra_info = extra
                
                self.session.load_data(raw, ds_name, sub, ctx)
                
                # Refresh all dependent tabs
                self.tab_info.refresh()
                self.tab_chans.refresh() # This will populate channel checkboxes
                self.tab_proc.refresh_options() # Updates available steps based on Raw/Epochs
                self.refresh_history()
                self.tabs.setCurrentIndex(0) # Switch to Info tab
            else:
                QMessageBox.warning(self, "Load Failed", "No data found for this selection.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def refresh_history(self):
        """Updates the history list widget."""
        self.list_history.clear()
        # Use the flat history from session state
        for step_display_text in self.session.get_history_flat():
            self.list_history.addItem(step_display_text)

    def delete_step_placeholder(self):
        QMessageBox.information(self, "Info", "Undo/Delete logic is complex (requires state stacking or re-processing). Not implemented yet.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIWorkstationGUI()
    window.show()
    sys.exit(app.exec())