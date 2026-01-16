import ast
import json
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QMessageBox, QFrame, QTabWidget, 
                             QListWidget, QCheckBox, QFileDialog, QGroupBox,QApplication,
                             QDialog, QDialogButtonBox)
from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from src.core.registry import Registry

class ProcessingTab(QWidget):
    """
    Main Preprocessing Widget.
    Layout:
    - Top (Persistent): Target Signal selection.
    - Center (Tabs): Build Pipeline vs Management/History.
    """
    stepApplied = pyqtSignal()

    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        self.inputs = {} 

        self._init_ui()


    def _init_ui(self):
        """Initializes the layout using QTabWidget with a Corner Widget."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. CREATE INTERNAL TAB SYSTEM
        self.internal_tabs = QTabWidget()
        
        # Sub-Tab A: Pipeline Builder
        self.tab_build = QWidget()
        self._setup_build_tab()
        
        # Sub-Tab B: History & Management
        self.tab_history = QWidget()
        self._setup_history_tab()

        self.internal_tabs.addTab(self.tab_build, "Configure Pipeline")
        self.internal_tabs.addTab(self.tab_history, "History & Management")

        # 2. CREATE CORNER WIDGET (The Focus Signal Switcher)
        corner_container = QWidget()
        h_corner = QHBoxLayout(corner_container)
        h_corner.setContentsMargins(10, 0, 10, 0)
        
        lbl_focus = QLabel("<b>Focus Signal:</b>")
        self.cb_target_signal = QComboBox()
        self.cb_target_signal.setFixedWidth(200)
        self.cb_target_signal.currentTextChanged.connect(self._on_context_switched)
        
        h_corner.addWidget(lbl_focus)
        h_corner.addWidget(self.cb_target_signal)
        
        # Set to Top Right of the tab bar
        self.internal_tabs.setCornerWidget(corner_container, Qt.Corner.TopRightCorner)
        
        self.main_layout.addWidget(self.internal_tabs)

    def _setup_build_tab(self):
        """UI for selecting functions and configuring parameters."""
        layout = QVBoxLayout(self.tab_build)
        
        # Selectors
        sel_frame = QFrame()
        sel_lay = QFormLayout(sel_frame)
        
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["raw_pipeline", "epoching", "epoch_pipeline"])
        self.combo_stage.currentTextChanged.connect(self.on_stage_changed)
        
        self.combo_steps = QComboBox()
        self.combo_steps.currentIndexChanged.connect(self.on_step_selected)
        
        sel_lay.addRow("Pipeline Stage:", self.combo_stage)
        sel_lay.addRow("Select Function:", self.combo_steps)
        layout.addWidget(sel_frame)

        # Dynamic Parameters Scroll
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.scroll.setWidget(self.form_container)
        layout.addWidget(self.scroll)

        # Broadcasting
        self.chk_apply_all = QCheckBox("Apply this step to ALL loaded signals in workspace")
        self.chk_apply_all.setStyleSheet("color: #d32f2f; font-weight: bold;")
        layout.addWidget(self.chk_apply_all)

        # Action
        self.btn_apply = QPushButton("Apply Step & Run")
        self.btn_apply.setFixedHeight(40)
        self.btn_apply.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_apply.clicked.connect(self.apply_step)
        layout.addWidget(self.btn_apply)

    def _setup_history_tab(self):
        """UI for viewing history and Export/Import JSON."""
        layout = QVBoxLayout(self.tab_history)
        
        self.list_history = QListWidget()
        layout.addWidget(QLabel("<b>Specific Pipeline History for Focus Signal:</b>"))
        layout.addWidget(self.list_history)

        # Undo/Reset
        h_btns = QHBoxLayout()
        btn_undo = QPushButton("Undo Last Step"); btn_undo.clicked.connect(self._undo)
        btn_reset = QPushButton("Reset Focused Signal"); btn_reset.setStyleSheet("color: red;")
        btn_reset.clicked.connect(self._reset)
        h_btns.addWidget(btn_undo); h_btns.addWidget(btn_reset)
        layout.addLayout(h_btns)

        # JSON IO
        grp_io = QGroupBox("Pipeline Transfer (JSON)")
        v_io = QVBoxLayout(grp_io)
        btn_import = QPushButton("Import JSON Pipeline")
        btn_export = QPushButton("Export Focused Pipeline to JSON")
        btn_import.clicked.connect(self._import_json)
        btn_export.clicked.connect(self._export_json)
        v_io.addWidget(btn_import); v_io.addWidget(btn_export)
        layout.addWidget(grp_io)

    # --- CORE SYNC LOGIC ---

    def refresh_options(self):
        """Synchronizes the persistent signal switcher with the session workspace."""
        if not self.session.workspace: return
        
        self.cb_target_signal.blockSignals(True)
        self.cb_target_signal.clear()
        self.cb_target_signal.addItems(self.session.logical_keys)
        
        if self.session.active_context:
            self.cb_target_signal.setCurrentText(self.session.active_context)
        self.cb_target_signal.blockSignals(False)

        self.on_stage_changed(self.combo_stage.currentText())
    

    def _on_context_switched(self, ctx_name):
        """Fired when switching signals in the persistent header."""
        if not ctx_name: return
        self.session.active_context = ctx_name
        self._refresh_history_list()
        # Update parameters form to show valid options for Raw vs Epochs
        self.on_stage_changed(self.combo_stage.currentText())
        # Refresh visuals in other tabs
        self.stepApplied.emit()

    def _refresh_history_list(self):
        """Populates the list with the pipeline steps of the focused context."""
        self.list_history.clear()
        for txt in self.session.get_history_flat():
            self.list_history.addItem(txt)

    # --- ACTION HANDLERS ---

    def apply_step(self):
        if not self.session.current_data: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.btn_apply.setEnabled(False)
        
        try:
            step_data = self.combo_steps.currentData()
            args = self._extract_form_data()
            if step_data['name'] == 'Epochs' and step_data['type'] == 'class_init':
                module_name = "mne"
            elif step_data['type'] == 'function':
                module_name = "src.preprocessing.preprocessing_functions"
            else:
                module_name = None
            config = {
                "type": step_data['type'],
                "name": step_data['name'],
                "module_name": module_name,
                "kwargs": args,
                "pipeline_stage": self.combo_stage.currentText()
            }

            
            success = self.session.run_transform(config, apply_to_all=self.chk_apply_all.isChecked())
            if success:
                self._refresh_history_list() 
                self.stepApplied.emit() # Habilitamos el timbre

            #     # --- SOLUCIÓN AL CRASH: REFRESCO DIFERIDO ---
            #     # QTimer.singleShot(10, self.stepApplied.emit)
            #     # QTimer.singleShot(20, self._refresh_history_list)
            #     # QTimer.singleShot(30, lambda: self.on_stage_changed(self.combo_stage.currentText()))
            # else:
            #     QMessageBox.warning(self, "Error", "Transform failed.")
                
        except Exception as e:
            print(f"[Processing] Error: {e}")
        finally:
            self.btn_apply.setEnabled(True)
            QApplication.restoreOverrideCursor()
            
    def _undo(self):
        if self.session.undo_last_step():
            self.stepApplied.emit()
            self._refresh_history_list()

    def _reset(self):
        if self.session.reset_pipeline(only_active=True):
            self.stepApplied.emit()
            self._refresh_history_list()

    def _export_json(self):
        """Exports all pipelines in the workspace and database metadata."""
        if not self.session.workspace: return

        # 1. Open a Dialog to get Metadata (Name/Description)
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Workspace Configuration")
        form = QFormLayout(dialog)
        le_name = QLineEdit("custom_pipeline")
        le_desc = QLineEdit("Created via BCI Workstation")
        form.addRow("Pipeline Name:", le_name)
        form.addRow("Description:", le_desc)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept); btns.rejected.connect(dialog.reject)
        form.addRow(btns)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            path, _ = QFileDialog.getSaveFileName(self, "Save JSON", f"{le_name.text()}.json", "JSON (*.json)")
            if not path: return

            # 1. DEEP COPY the pipelines to avoid modifying the live session state
            import copy
            cleaned_pipelines = copy.deepcopy(self.session.pipelines)

            # 2. SANITIZE: Remove large NumPy arrays from the 'epoching' step
            for ctx_name, pipe in cleaned_pipelines.items():
                epoch_step = pipe.get('epoching')
                if epoch_step and 'kwargs' in epoch_step:
                    epoch_step['kwargs']['events'] = None

            # 3. Build the structure
            export_data = {
                "preprocessing_name": le_name.text(),
                "description": le_desc.text(),
                "database": {
                    "class_name": self.session.dataset_name,
                    "module_name": "src.eeg_datasets", 
                    "kwargs": {"data_to_load": list(self.session.workspace.keys())}
                },
                "preprocessing_pipeline": cleaned_pipelines 
            }

            try:
                with open(path, 'w') as f:
                    json.dump(export_data, f, indent=4)
                QMessageBox.information(self, "Success", "Cleaned pipeline exported.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed: {e}")
                
    def _import_json(self):
        """Imports pipelines and maps them to existing workspace contexts."""
        path, _ = QFileDialog.getOpenFileName(self, "Import JSON", "", "JSON (*.json)")
        if not path: return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Identify where the pipeline dictionary is
            pipeline_dict = data.get('preprocessing_pipeline', data)

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            # Apply using the new Session method
            success = self.session.load_workspace_pipelines_from_json(pipeline_dict)
            
            QApplication.restoreOverrideCursor()

            if success:
                self.stepApplied.emit()
                self._refresh_history_list()
                QMessageBox.information(self, "Import Successful", "Workspace pipelines updated for available contexts.")
            else:
                QMessageBox.warning(self, "Import Partial", "Some steps failed. Check logs.")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Could not parse JSON: {e}")

    # --- REUSED HELPERS (Registry/Form) ---

    def on_stage_changed(self, stage_name):
        if not self.session.current_data: return
        self.combo_steps.blockSignals(True); self.combo_steps.clear()
        if stage_name == "epoching":
            self.combo_steps.addItem("mne.Epochs (Transition)", userData={"name": "Epochs", "type": "class_init"})
            self.combo_steps.setEnabled(False); self._build_form("Epochs", "class_init")
        else:
            self.combo_steps.setEnabled(True)
            avail_steps = Registry.get_available_steps(self.session.current_data)
            for step in avail_steps:
                if step['name'] in ['Epochs', 'convert_to_epochs']: continue
                self.combo_steps.addItem(f"{step['name']} ({step['type']})", userData=step)
            if self.combo_steps.count() > 0: 
                self.combo_steps.setCurrentIndex(0)
                step_data = self.combo_steps.itemData(0)
                self._build_form(step_data['name'], step_data['type'])
            else:
                self._clear_form()
        self.combo_steps.blockSignals(False)

    def on_step_selected(self):
        data = self.combo_steps.currentData()
        if data: self._build_form(data['name'], data['type'])

    def _build_form(self, name, step_type):
        self._clear_form()
        defaults = Registry.get_defaults(name, step_type)
        if not defaults: return
        for arg, val in defaults.items():
            if arg in ['self', 'raw', 'events', 'epochs', 'data', 'inst']: continue
            le = QLineEdit(str(val) if val is not None else "None")
            self.form_layout.addRow(QLabel(str(arg)), le); self.inputs[arg] = le

    def _clear_form(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}

    def _extract_form_data(self):
        data = {}
        for arg, le in self.inputs.items():
            txt = le.text().strip()
            try: val = ast.literal_eval(txt)
            except: val = txt
            data[arg] = val
        if 'kwargs' in data:
            extra = data.pop('kwargs')
            if isinstance(extra, dict): data.update(extra)
        return data