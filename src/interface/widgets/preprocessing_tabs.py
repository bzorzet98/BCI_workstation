import ast
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QMessageBox, QFrame)
from PyQt6.QtCore import pyqtSignal, Qt
from src.core.registry import Registry

class ProcessingTab(QWidget):
    stepApplied = pyqtSignal()

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.inputs = {} # Stores QLineEdit objects for the dynamic form
        
        self._init_ui()

    def _init_ui(self):
        """Initializes the layout structure."""
        layout = QVBoxLayout(self)

        # 1. HEADER: Stage & Step Selection
        selection_frame = QFrame()
        sel_layout = QVBoxLayout(selection_frame)
        
        # Target Stage
        h_stage = QHBoxLayout()
        h_stage.addWidget(QLabel("<b>Target Stage:</b>"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["raw_pipeline", "epoching", "epoch_pipeline"])
        self.combo_stage.currentTextChanged.connect(self.on_stage_changed)
        h_stage.addWidget(self.combo_stage)
        sel_layout.addLayout(h_stage)

        # Function/Method Selector
        sel_layout.addWidget(QLabel("<b>Select Function/Method:</b>"))
        self.combo_steps = QComboBox()
        self.combo_steps.currentIndexChanged.connect(self.on_step_selected)
        sel_layout.addWidget(self.combo_steps)
        
        layout.addWidget(selection_frame)

        # 2. BODY: Dynamic Form (Scrollable)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.scroll.setWidget(self.form_container)
        
        layout.addWidget(self.scroll)

        # 3. FOOTER: Apply Action
        self.btn_apply = QPushButton("Add Step & Run")
        self.btn_apply.setFixedHeight(40)
        self.btn_apply.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.btn_apply.clicked.connect(self.apply_step)
        layout.addWidget(self.btn_apply)

    # --- PUBLIC API ---

    def refresh_options(self):
        """Called by the main controller when data is loaded."""
        self.on_stage_changed(self.combo_stage.currentText())

    # --- EVENT HANDLERS ---

    def on_stage_changed(self, stage_name):
        """Handles logic when switching between pipeline stages."""
        if not self.session.current_data: return

        self.combo_steps.blockSignals(True)
        self.combo_steps.clear()

        if stage_name == "epoching":
            # --- TRANSITION TO EPOCHS ---
            epoch_data = {"name": "Epochs", "type": "class_init"}
            self.combo_steps.addItem("mne.Epochs (Transition)", userData=epoch_data)
            self.combo_steps.setEnabled(False) 
            self._build_form("Epochs", "class_init")
        else:
            # --- STANDARD METHODS ---
            self.combo_steps.setEnabled(True)
            avail_steps = Registry.get_available_steps(self.session.current_data)
            
            for step in avail_steps:
                # Hide transition methods from standard selection
                if step['name'] in ['Epochs', 'convert_to_epochs']: continue
                self.combo_steps.addItem(f"{step['name']} ({step['type']})", userData=step)
            
            if self.combo_steps.count() > 0:
                self.on_step_selected()
            else:
                self._clear_form()

        self.combo_steps.blockSignals(False)

    def on_step_selected(self):
        """Triggers form rebuild when the user selects a different method."""
        step_data = self.combo_steps.currentData()
        if step_data:
            self._build_form(step_data['name'], step_data['type'])

    # --- UI HELPERS ---

    def _build_form(self, name, step_type):
        """Fetches arguments from Registry and creates input fields."""
        self._clear_form()
        defaults = Registry.get_defaults(name, step_type)
        
        if not defaults:
            msg = "Epochs parameters error" if step_type == 'class_init' else "No arguments required."
            self.form_layout.addRow(QLabel(msg), QLabel(""))
            return

        for arg, val in defaults.items():
            # Skip internal MNE/Python arguments
            if arg in ['self', 'raw', 'events', 'epochs', 'data', 'inst']: continue
            
            lbl = QLabel(str(arg))
            # Convert None to "None" string for the UI
            display_val = str(val) if val is not None else "None"
            le = QLineEdit(display_val)
            le.setStyleSheet("background-color: white; border: 1px solid #ced4da; padding: 4px;")
            
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def _clear_form(self):
        """Safely removes all widgets from the dynamic form."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): 
                item.widget().deleteLater()
        self.inputs = {}

    def _extract_form_data(self):
        """Parses UI inputs and flattens 'kwargs' dictionaries."""
        data = {}
        for arg, le in self.inputs.items():
            txt = le.text().strip()
            try: 
                # Convert strings like "[1, 2]" or "0.5" into Python types
                val = ast.literal_eval(txt)
            except: 
                val = txt # Fallback to raw string
            data[arg] = val
            
        # Handle Nested Kwargs: If the user provided a dict in the 'kwargs' field, flatten it
        if 'kwargs' in data:
            extra = data.pop('kwargs')
            if isinstance(extra, dict):
                data.update(extra)
        
        return data

    # --- CONTROLLER LOGIC ---

    def apply_step(self):
        """Constructs the step configuration and sends it to the SessionState."""
        if not self.session.current_data: return
        
        # 1. Prepare Data
        args = self._extract_form_data()
        step_data = self.combo_steps.currentData()
        
        # 2. Build Pipeline Configuration
        is_custom = step_data['type'] == 'function'
        config = {
            "type": step_data['type'],
            "name": step_data['name'],
            "module_name": "src.preprocessing.preprocessing_functions" if is_custom else None,
            "kwargs": args,
            "pipeline_stage": self.combo_stage.currentText()
        }
        
        # 3. Execute and Notify
        if self.session.run_transform(config):
            self.stepApplied.emit()
            self.refresh_options()
        else:
            QMessageBox.warning(self, "Transform Failed", f"Could not apply {step_data['name']}.")