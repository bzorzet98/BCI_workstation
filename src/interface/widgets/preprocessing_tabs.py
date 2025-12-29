import sys
import ast
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QMessageBox)
from PyQt6.QtCore import pyqtSignal
from src.core.registry import Registry

class ProcessingTab(QWidget):
    stepApplied = pyqtSignal()

    def __init__(self, session):
        super().__init__()
        self.session = session
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. Pipeline Stage Selector
        h_stage = QHBoxLayout()
        h_stage.addWidget(QLabel("Target Stage:"))
        self.combo_stage = QComboBox()
        self.combo_stage.addItems(["raw_pipeline", "epoching", "epoch_pipeline"])
        self.combo_stage.currentTextChanged.connect(self.on_stage_changed)
        h_stage.addWidget(self.combo_stage)
        layout.addLayout(h_stage)

        # 2. Step Selector
        layout.addWidget(QLabel("Select Function/Method:"))
        self.combo_steps = QComboBox()
        self.combo_steps.currentIndexChanged.connect(self.on_step_selected)
        layout.addWidget(self.combo_steps)

        # 3. Dynamic Form
        self.form_widget = QWidget()
        self.form_layout = QFormLayout()
        self.form_widget.setLayout(self.form_layout)
        scroll = QScrollArea()
        scroll.setWidget(self.form_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        self.inputs = {}

        # 4. Apply Button
        self.btn_apply = QPushButton("Add Step & Run")
        self.btn_apply.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_apply.clicked.connect(self.apply_step)
        layout.addWidget(self.btn_apply)

    def refresh_options(self):
        """Called when data is loaded."""
        self.on_stage_changed(self.combo_stage.currentText())

    def on_stage_changed(self, stage_name):
        """
        Handles logic when switching between Raw, Epoching, and Post-Epoch stages.
        """
        if not self.session.current_data: return

        self.combo_steps.blockSignals(True)
        self.combo_steps.clear()

        if stage_name == "epoching":
            # --- EPOCHING MODE ---
            # 1. Add the specific visual item
            # We explicitly set the type to 'class_init' so Registry knows what to do
            epoch_data = {"name": "Epochs", "type": "class_init"}
            self.combo_steps.addItem("mne.Epochs (Transition)", userData=epoch_data)
            self.combo_steps.setCurrentIndex(0)
            self.combo_steps.setEnabled(False) 
            
            # 2. Build form immediately
            # We pass the dictionary we just created
            self.build_form_for_step(epoch_data['name'], epoch_data['type'])

        else:
            # --- STANDARD MODE ---
            self.combo_steps.setEnabled(True)
            avail_steps = Registry.get_available_steps(self.session.current_data)
            
            for step in avail_steps:
                # Hide epoching wrappers from standard pipelines
                if step['name'] in ['Epochs', 'convert_to_epochs']: continue
                self.combo_steps.addItem(f"{step['name']} ({step['type']})", userData=step)
            
            # Trigger first item
            if self.combo_steps.count() > 0:
                step_data = self.combo_steps.itemData(0)
                self.build_form_for_step(step_data['name'], step_data['type'])
            else:
                self.clear_form()

        self.combo_steps.blockSignals(False)

    def on_step_selected(self):
        """Standard dropdown selection."""
        idx = self.combo_steps.currentIndex()
        if idx < 0: return
        step_data = self.combo_steps.currentData()
        self.build_form_for_step(step_data['name'], step_data['type'])

    def build_form_for_step(self, name, step_type):
        """Fetches defaults from Registry and creates input fields."""
        self.clear_form()
        
        # DEBUG: Print to console to verify Registry is working
        # print(f"Building form for: {name} ({step_type})")
        
        defaults = Registry.get_defaults(name, step_type)
        
        if not defaults:
            # Fallback label if no params found (e.g. function takes no args)
            if step_type == 'class_init':
                self.form_layout.addRow(QLabel("Error: Could not retrieve Epochs parameters."), QLabel(""))
            else:
                self.form_layout.addRow(QLabel("No arguments."), QLabel(""))
            return

        for arg, val in defaults.items():
            # Filter internal args
            if arg in ['self', 'raw', 'events', 'epochs', 'data', 'inst']: continue
            
            lbl = QLabel(str(arg))
            # Handle None representation
            display_val = str(val) if val is not None else "None"
            le = QLineEdit(display_val)
            
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def clear_form(self):
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}

    def build_form_for_step(self, name, step_type):
        """Fetches defaults from Registry and creates input fields."""
        self.clear_form()
        
        # Registry look up
        # Note: If name='Epochs' and step_type='class_init' (or 'class'), Registry handles __init__ logic
        defaults = Registry.get_defaults(name, step_type)
        
        for arg, val in defaults.items():
            # Skip internal arguments
            if arg in ['self', 'raw', 'events', 'epochs', 'data']: continue
            
            lbl = QLabel(str(arg))
            le = QLineEdit(str(val))
            
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def apply_step(self):
        if not self.session.current_data: return
        
        # 1. Parse Arguments
        kwargs = {}
        for arg, le in self.inputs.items():
            txt = le.text()
            try: kwargs[arg] = ast.literal_eval(txt)
            except: kwargs[arg] = txt
            
        target_stage = self.combo_stage.currentText()
        step_data = self.combo_steps.currentData()
        
        # 2. Construct Configuration
        if target_stage == "epoching":
            # FORCE correct config for Worker
            step_config = {
                "type": "class",
                "name": "Epochs",
                "module_name": "mne",
                "kwargs": kwargs,
                "pipeline_stage": "epoching"
            }
        else:
            # Standard config
            step_config = {
                "type": step_data['type'],
                "name": step_data['name'],
                "module_name": "src.preprocessing.preprocessing_functions" if step_data['type']=='function' else None,
                "kwargs": kwargs,
                "pipeline_stage": target_stage
            }
        
        # 3. Execute
        success = self.session.run_transform(step_config)
        if success:
            self.stepApplied.emit()
            self.refresh_options() # Refresh in case type changed (Raw -> Epochs)
        else:
            QMessageBox.warning(self, "Error", "Transform failed.")