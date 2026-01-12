# PyQt Imports
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout,  QComboBox, QPushButton, 
                             QSpinBox, QListWidget, QGroupBox, 
                             QFormLayout, QMessageBox, QTabWidget,
                             QFileDialog, QDialog, QDialogButtonBox, QLabel, QLineEdit, QSizePolicy,
                             QApplication) # Added QSizePolicy for canvas
from PyQt6.QtCore import QTimer ,Qt
from PyQt6.QtGui import QPixmap
import os
# --- CORE IMPORTS ---
from src.core.data_manager import DataManager
from src.core.session import SessionState

# --- WIDGET IMPORTS ---
# Make sure these paths are correct based on your file structure
from src.interface.widgets.channel_tab import ChannelTab
from src.interface.widgets.info_tab import DatasetInfoTab # Renamed
from src.interface.widgets.metadata_tab import SubjectMetadataTab # New
from src.interface.widgets.freq_tab import FreqDomainTab # New
from src.interface.widgets.preprocessing_tabs import ProcessingTab

from src.interface.widgets.log_console import LogConsole
from src.interface.widgets.time_tab import TimeDomainTab
from src.interface.widgets.eda_tab import EDATab 

# --- MAIN WINDOW ---
# Add LogConsole if you implemented it
# from src.interface.widgets.log_console import LogConsole

class BCIWorkstationGUI(QMainWindow):
    def __init__(self,style_kwargs=None):
        super().__init__()
        self.setWindowTitle("BCI Workstation Professional")
        self.resize(1200, 800)
        
        # --- MODEL ---
        self.data_manager = DataManager()
        self.session = SessionState()

        # --- MAIN LAYOUT STRUCTURE ---
        central = QWidget()
        self.setCentralWidget(central)
        
        # The main layout is VERTICAL: Top (App) / Bottom (Logs)
        main_layout = QVBoxLayout(central)

        # ==========================================
        # TOP AREA: Sidebar (Left) + Tabs (Right)
        # ==========================================
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0) # Remove padding for cleaner look
        
        # --- LEFT SIDEBAR ---
        sidebar_layout = QVBoxLayout()
        w_side = QWidget()
        w_side.setFixedWidth(320)
        w_side.setLayout(sidebar_layout)
        
        # --- 0. LOGO PLACEMENT (NEW) ---
        self.logo_label = QLabel()
        if style_kwargs is None:
            style_kwargs = {}
        else:
            pyplot_style = style_kwargs.get("pyplot_style", None)
            logo_path = style_kwargs.get("logo_path", None)
        self.style_kwargs = pyplot_style
        try:
            # Replace 'assets/lab_logo.png' with your actual path
            # Creating a dummy pixmap if you don't have one yet to verify layout
            logo_pixmap = QPixmap(logo_path)
            
            if logo_pixmap.isNull():
                # Fallback text if image not found
                self.logo_label.setText("<b>BCI WORKSTATION</b>")
                self.logo_label.setStyleSheet("font-size: 24px; color: #589df6; margin-bottom: 20px;")
            else:
                scaled_pixmap = logo_pixmap.scaled(280, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled_pixmap)
                self.logo_label.setStyleSheet("margin-bottom: 20px; border: none;")
                
        except Exception:
             self.logo_label.setText("<b>BCI WORKSTATION</b>")
        
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add logo to layout FIRST
        sidebar_layout.addWidget(self.logo_label)

        # 1. Data Selection
        grp_load = QGroupBox("Data Loading")
        form_load = QFormLayout()
        
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(self.data_manager.get_available_datasets())
        self.cb_dataset.currentTextChanged.connect(self.on_dataset_change)
        
        self.sb_subject = QSpinBox()
        self.sb_subject.setRange(1, 100)
        
        self.cb_context = QComboBox()
        
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.load_data)
        
        form_load.addRow("Dataset:", self.cb_dataset)
        form_load.addRow("Subject:", self.sb_subject)
        form_load.addRow("Context:", self.cb_context)
        form_load.addRow(btn_load)
        grp_load.setLayout(form_load)
        sidebar_layout.addWidget(grp_load)

        # 2. Pipeline History
        grp_hist = QGroupBox("Pipeline History")
        v_hist = QVBoxLayout()
        self.list_history = QListWidget()
        v_hist.addWidget(self.list_history)
        
        # Action Buttons
        h_btns = QHBoxLayout()
        btn_undo = QPushButton("Undo Last")
        btn_undo.clicked.connect(self.undo_step)
        btn_reset = QPushButton("Reset All")
        btn_reset.setStyleSheet("color: red;")
        btn_reset.clicked.connect(self.reset_pipeline)
        
        h_btns.addWidget(btn_undo)
        h_btns.addWidget(btn_reset)
        v_hist.addLayout(h_btns)
        
        grp_hist.setLayout(v_hist)
        sidebar_layout.addWidget(grp_hist)
        
        # Add Export Button
        btn_export = QPushButton("Export Pipeline JSON")
        btn_export.clicked.connect(self.export_pipeline_dialog)
        sidebar_layout.addWidget(btn_export)
        
        btn_import = QPushButton("Import & Run Pipeline")
        btn_import.clicked.connect(self.import_pipeline_dialog)
        sidebar_layout.addWidget(btn_import)

        # --- RIGHT TABS ---
        self.tabs = QTabWidget()
        
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Instantiate Tabs
        self.tab_info = DatasetInfoTab(self.session, self.data_manager)
        self.tab_meta = SubjectMetadataTab(self.session, self.data_manager, self.style_kwargs)
        self.tab_chans = ChannelTab(self.session)
        self.tab_proc = ProcessingTab(self.session)
        self.tab_time = TimeDomainTab(self.session, self.style_kwargs)
        self.tab_freq = FreqDomainTab(self.session, self.style_kwargs) # New
        self.tab_eda = EDATab(self.session, self.style_kwargs) # New
        
        # Add Tabs
        self.tabs.addTab(self.tab_info, "Dataset Info")
        self.tabs.addTab(self.tab_meta, "Demographics")
        self.tabs.addTab(self.tab_chans, "Channels")
        self.tabs.addTab(self.tab_proc, "Preprocessing")
        self.tabs.addTab(self.tab_time, "Time Domain")
        self.tabs.addTab(self.tab_freq, "Frequency Domain")
        self.tabs.addTab(self.tab_eda, "EDA Diagnostic") # New

        self.tab_proc.stepApplied.connect(self.refresh_gui)

        # Add Sidebar and Tabs to Top Layout
        top_layout.addWidget(w_side)
        top_layout.addWidget(self.tabs)
        
        # Add Top Widget to Main Layout (Stretch factor 3 means it takes 75% height)
        main_layout.addWidget(top_widget, stretch=3)

        # ==========================================
        # BOTTOM AREA: Log Console
        # ==========================================
        grp_console = QGroupBox("Execution Log")
        grp_console.setMinimumHeight(150) 
        v_cons = QVBoxLayout()
        self.console = LogConsole() # Requires src/interface/widgets/log_console.py
        v_cons.addWidget(self.console)
        grp_console.setLayout(v_cons)
        
        # Add Console to Main Layout (Stretch factor 1 means it takes 25% height)
        main_layout.addWidget(grp_console, stretch=1)

        # Initial trigger
        if self.cb_dataset.count() > 0:
            QTimer.singleShot(0, lambda: self.on_dataset_change(self.cb_dataset.currentText()))
    # --- LOGIC ---

    def on_dataset_change(self, name):
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
            print(f"Error loading meta: {e}")

    def load_data(self):
        try:
            raw = self.data_manager.load_data(self.sb_subject.value(), self.cb_context.currentText())
            if raw:
                # Fetch extra info
                full_data = self.data_manager.active_dataset_instance.get_data(subjects=[self.sb_subject.value()])
                extra = full_data[self.sb_subject.value()].get('extra_info', {})
                self.session.extra_info = extra
                
                self.session.load_data(raw, self.cb_dataset.currentText(), self.sb_subject.value(), self.cb_context.currentText())

                self.refresh_gui()
                self.tabs.setCurrentIndex(0)
                
                # If using specialized tab:
                self.tab_time.refresh()
            else:
                QMessageBox.warning(self, "Load Failed", "No data found.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def refresh_gui(self):
        """Central method to update all UI elements after a data change."""
        self.refresh_history()
        # Refresh the new split tabs
        self.tab_info.refresh()
        self.tab_meta.refresh()
        self.tab_chans.refresh()
        self.tab_proc.refresh_options()
        self.tab_time.refresh()
        self.tab_freq.refresh()

    def refresh_history(self):
        self.list_history.clear()
        for txt in self.session.get_history_flat():
            self.list_history.addItem(txt)

    def undo_step(self):
        if self.session.undo_last_step():
            self.refresh_gui()
        else:
            QMessageBox.information(self, "Info", "Nothing to undo.")

    def reset_pipeline(self):
        if self.session.reset_pipeline():
            self.refresh_gui()
            QMessageBox.information(self, "Info", "Pipeline reset to original data.")
            
    def export_pipeline_dialog(self):
        """
        Opens a dialog to get metadata and saves the pipeline JSON.
        """
        # 1. Custom Dialog for Metadata
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Configuration")
        layout = QFormLayout(dialog)
        
        le_name = QLineEdit("custom_pipeline")
        le_desc = QLineEdit("Created via BCI Workstation")
        
        layout.addRow("Preprocessing Name:", le_name)
        layout.addRow("Description:", le_desc)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            prep_name = le_name.text()
            desc = le_desc.text()
            self.save_pipeline_json(prep_name, desc)
            
    def import_pipeline_dialog(self):
        """Opens file dialog to load a pipeline JSON."""
        if not self.session.current_data:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pipeline JSON",
            "", # Current dir
            "JSON Files (*.json)"
        )
        
        if filename:
            # Show a loading cursor because this might take time
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            success = self.session.load_pipeline_from_json(filename)
            
            QApplication.restoreOverrideCursor()
            
            if success:
                self.refresh_gui()
                QMessageBox.information(self, "Success", "Pipeline applied successfully.")
            else:
                QMessageBox.critical(self, "Error", "Pipeline execution failed. Check logs.")

    def save_pipeline_json(self, name, description):
        """Constructs the final JSON and saves to file."""
        
        # 1. Build the structure
        export_data = {
            "preprocessing_name": name,
            "description": description,
            "database": {
                "class_name": self.cb_dataset.currentText(),
                "module_name": "src.eeg_datasets", # Standard assumption
                "kwargs": {
                    "data_to_load": [self.cb_context.currentText()]
                }
            },
            "preprocessing_pipeline": self.session.pipeline # This is already structured correctly
        }
        
        # 2. File Dialog
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Pipeline JSON", 
            f"{name}.json", 
            "JSON Files (*.json)"
        )
        
        if filename:
            try:
                import json
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=4)
                QMessageBox.information(self, "Success", f"Pipeline saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")