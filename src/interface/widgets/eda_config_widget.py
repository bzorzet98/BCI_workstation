import inspect
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QStackedWidget, QListWidget, QGroupBox, QMessageBox)
from PyQt6.QtCore import pyqtSignal

from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox  # Assuming you have this
import ast

class EDAConfigWidget(QWidget):
    
    analysis_requested = pyqtSignal(str, dict, list) 

    def __init__(self, session_state, parent=None):
        super().__init__(parent)
        self.session = session_state
        self.analyzer_bundles = {} # Stores UI inputs for each analyzer
        
        self._build_ui()
        self._prebuild_analyzer_uis()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top: Configuration Panel ---
        group_config = QGroupBox("EDA Configuration")
        v_config = QVBoxLayout(group_config)

        # Context Selection (Which signals to analyze)
        self.combo_contexts = CheckableComboBox()
        v_config.addWidget(QLabel("Select Contexts:"))
        v_config.addWidget(self.combo_contexts)

        # Analyzer Selection
        self.combo_analyzer = QComboBox()
        self.combo_analyzer.currentIndexChanged.connect(self._on_analyzer_changed)
        v_config.addWidget(QLabel("Select Analyzer:"))
        v_config.addWidget(self.combo_analyzer)

        # Stacked Widget for dynamic parameters
        self.stack_params = QStackedWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.stack_params)
        v_config.addWidget(scroll)

        # Apply Button
        self.btn_apply = QPushButton("Apply Analysis")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        v_config.addWidget(self.btn_apply)

        main_layout.addWidget(group_config, stretch=2)

        # --- Bottom: History Panel ---
        group_history = QGroupBox("Analysis History")
        v_hist = QVBoxLayout(group_history)
        self.list_history = QListWidget()
        # You can add double-click connections here later to reload history
        v_hist.addWidget(self.list_history)
        main_layout.addWidget(group_history, stretch=1)

    def _prebuild_analyzer_uis(self):
        """Dynamically builds the UI parameters based on analyzer __init__ and analyze() signatures."""
        analyzers = Registry.discover_eda_analyzers()
        
        for name, path in analyzers.items():
            self.combo_analyzer.addItem(name)
            page = QWidget()
            layout = QFormLayout(page)
            inputs = {}

            # Get __init__ params (Standard parameters like tmin, z_threshold)
            init_params = Registry.get_analyzer_params(name, path)
            for p_name, p_val in init_params.items():
                if p_name == 'kwargs': continue
                le = QLineEdit(str(p_val) if p_val is not None else "")
                layout.addRow(QLabel(f"{p_name.replace('_', ' ').title()}:"), le)
                inputs[p_name] = le

            # Get analyze() extra params (Like reference, baseline)
            Cls = import_class(name, path)
            analyze_sig = inspect.signature(Cls.analyze)
            params = [p for p in analyze_sig.parameters.values() if p.name not in ['self', 'data', 'inst']]
            
            for extra in params:
                label_text = f"<b>{extra.name.replace('_',' ').title()} (Context):</b>"
                cb_extra = QComboBox() # Or CheckableComboBox if it accepts multiple
                # We will populate these items dynamically when refresh() is called
                layout.addRow(QLabel(label_text), cb_extra)
                inputs[extra.name] = cb_extra

            self.analyzer_bundles[name] = {'page': page, 'inputs': inputs, 'extra_params': [p.name for p in params]}
            self.stack_params.addWidget(page)

    def _on_analyzer_changed(self):
        name = self.combo_analyzer.currentText()
        if name in self.analyzer_bundles:
            self.stack_params.setCurrentWidget(self.analyzer_bundles[name]['page'])

    def refresh_contexts(self):
        """Called by the main GUI when data is loaded/changed to populate combo boxes."""
        contexts = list(self.session.workspace.keys())
        self.combo_contexts.clear()
        self.combo_contexts.addItems(contexts)
        
        # Update dynamic context dropdowns (like 'reference') across all stacked widgets
        for bundle in self.analyzer_bundles.values():
            for extra_name in bundle['extra_params']:
                cb = bundle['inputs'][extra_name]
                cb.clear()
                cb.addItems(["None"] + contexts)
                
    def _on_apply_clicked(self):
        """Gathers parameters and emits them cleanly separated for Init and Runtime."""
        name = self.combo_analyzer.currentText()
        
        # --- ROBUST CONTEXT EXTRACTION ---
        contexts = []
        if hasattr(self.combo_contexts, 'get_checked_items'):
            contexts = self.combo_contexts.get_checked_items()
        elif hasattr(self.combo_contexts, 'currentData') and isinstance(self.combo_contexts.currentData(), list):
            contexts = self.combo_contexts.currentData()
        else:
            from PyQt6.QtCore import Qt
            for i in range(self.combo_contexts.count()):
                state = self.combo_contexts.itemData(i, Qt.ItemDataRole.CheckStateRole)
                if state == Qt.CheckState.Checked or state == 2:
                    contexts.append(self.combo_contexts.itemText(i))

        if not contexts:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Warning", "Please select at least one context to analyze.")
            return

        bundle = self.analyzer_bundles[name]
        extra_param_keys = bundle.get('extra_params', [])
        
        init_kwargs = {}
        call_kwargs = {}

        # Safely extract all values from the UI
        import ast
        for k, widget in bundle['inputs'].items():
            val = None
            
            # Extract from QLineEdit
            if isinstance(widget, QLineEdit):
                raw_val = widget.text().strip()
                if raw_val:
                    try:
                        val = ast.literal_eval(raw_val)
                    except (ValueError, SyntaxError):
                        try: 
                            val = float(raw_val) if '.' in raw_val else int(raw_val)
                        except ValueError: 
                            val = raw_val
                            
            # Extract from QComboBox
            elif isinstance(widget, QComboBox):
                raw_val = widget.currentText()
                val = raw_val if raw_val != "None" else None

            # --- SMART ROUTING ---
            # If the parameter is an 'extra_param' (like reference/baseline), it goes to call_kwargs
            if k in extra_param_keys:
                call_kwargs[k] = val
            # Otherwise, it's a standard math parameter for __init__
            else:
                init_kwargs[k] = val

        # --- THE SMART PAYLOAD ---
        payload = {
            "init_kwargs": init_kwargs,
            "call_kwargs": call_kwargs
        }

        # YELL OUT THE SIGNAL!
        self.analysis_requested.emit(name, payload, contexts)
        
    def add_history_entry(self, analyzer_name):
        self.list_history.addItem(f"{analyzer_name} #{self.list_history.count() + 1}")