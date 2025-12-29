import sys
import ast
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QSpinBox, QListWidget, QGroupBox, QLineEdit, 
                             QFormLayout, QMessageBox, QTabWidget, QScrollArea)
from PyQt6.QtCore import Qt

# Import your core logic
from src.core.data_manager import DataManager
from src.core.session import SessionState
from src.core.registry import Registry
from src.interface.plotter import Plotter

class DynamicStepForm(QWidget):
    """
    A widget that dynamically generates input fields based on function arguments.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout()
        self.setLayout(self.layout)
        self.inputs = {}

    def clear(self):
        # Remove all rows
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        self.inputs = {}

    def build_form(self, defaults):
        self.clear()
        for arg, val in defaults.items():
            # Create label
            label = QLabel(str(arg))
            
            # Create input field
            # We use QLineEdit for flexibility (parsing strings/lists)
            line_edit = QLineEdit()
            line_edit.setText(str(val))
            
            self.layout.addRow(label, line_edit)
            self.inputs[arg] = line_edit

    def get_values(self):
        """Parses current text in fields back to Python types."""
        values = {}
        for arg, widget in self.inputs.items():
            text = widget.text().strip()
            # Basic type inference similar to CLI
            try:
                # Try parsing literal (list, bool, number)
                val = ast.literal_eval(text)
            except:
                # Fallback to string
                val = text
            values[arg] = val
        return values

class BCIWorkstationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BCI Workstation GUI")
        self.resize(1000, 700)

        # --- MODEL INITIALIZATION ---
        self.data_manager = DataManager()
        self.session = SessionState()

        # --- LAYOUT SETUP ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. LEFT SIDEBAR (Load & Pipeline)
        sidebar = QVBoxLayout()
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(300)
        sidebar_widget.setLayout(sidebar)
        main_layout.addWidget(sidebar_widget)

        # --- Section A: Data Loading ---
        grp_load = QGroupBox("1. Data Loading")
        form_load = QFormLayout()
        
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(self.data_manager.get_available_datasets())
        self.combo_dataset.currentTextChanged.connect(self.on_dataset_changed)
        
        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(self.data_manager.get_available_datasets())
        self.combo_dataset.currentTextChanged.connect(self.on_dataset_changed)
    
        self.spin_sub = QSpinBox()
        self.spin_sub.setRange(1, 100)
        
        self.combo_context = QComboBox()
        if self.combo_dataset.count() > 0:
            self.on_dataset_changed(self.combo_dataset.currentText())
            
        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_data)
        
        form_load.addRow("Dataset:", self.combo_dataset)
        form_load.addRow("Subject:", self.spin_sub)
        form_load.addRow("Context:", self.combo_context)
        form_load.addRow(self.btn_load)
        grp_load.setLayout(form_load)
        sidebar.addWidget(grp_load)

        # --- Section B: Pipeline Steps (History) ---
        grp_hist = QGroupBox("2. Pipeline History")
        vbox_hist = QVBoxLayout()
        self.list_history = QListWidget()
        vbox_hist.addWidget(self.list_history)
        grp_hist.setLayout(vbox_hist)
        sidebar.addWidget(grp_hist)

        # 2. RIGHT PANEL (Actions)
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)

        # --- Status Bar ---
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setStyleSheet("font-weight: bold; color: blue; font-size: 14px;")
        right_panel.addWidget(self.lbl_status)

        # --- Action Tabs ---
        self.tabs = QTabWidget()
        right_panel.addWidget(self.tabs)

        # Tab 1: Processing
        self.tab_process = QWidget()
        self.setup_processing_tab()
        self.tabs.addTab(self.tab_process, "Preprocessing")

        # Tab 2: Visualization
        self.tab_viz = QWidget()
        self.setup_viz_tab()
        self.tabs.addTab(self.tab_viz, "Visualization")

        # Trigger initial dataset load to populate contexts
        self.on_dataset_changed(self.combo_dataset.currentText())

    def setup_processing_tab(self):
        layout = QVBoxLayout()
        self.tab_process.setLayout(layout)

        # Selector
        layout.addWidget(QLabel("Select Processing Step:"))
        self.combo_steps = QComboBox()
        self.combo_steps.currentIndexChanged.connect(self.on_step_selected)
        layout.addWidget(self.combo_steps)

        # Dynamic Form Container
        self.step_form = DynamicStepForm()
        scroll = QScrollArea()
        scroll.setWidget(self.step_form)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Apply Button
        self.btn_apply = QPushButton("Apply Step")
        self.btn_apply.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_apply.clicked.connect(self.apply_step)
        layout.addWidget(self.btn_apply)

    def setup_viz_tab(self):
        layout = QVBoxLayout()
        self.tab_viz.setLayout(layout)
        
        btn_plot = QPushButton("Interactive Plot (Time)")
        btn_plot.clicked.connect(lambda: Plotter.plot_interactive(self.session.current_data))
        
        btn_psd = QPushButton("Plot PSD (Frequency)")
        btn_psd.clicked.connect(lambda: Plotter.plot_psd(self.session.current_data))
        
        btn_sensors = QPushButton("Plot Sensors")
        btn_sensors.clicked.connect(lambda: Plotter.plot_sensors(self.session.current_data))

        layout.addWidget(btn_plot)
        layout.addWidget(btn_psd)
        layout.addWidget(btn_sensors)
        layout.addStretch()

    # --- LOGIC SLOTS ---

    def on_dataset_changed(self, ds_name):
        """
        Triggered when the user selects a different dataset.
        It reloads the dataset class and updates the Contexts and Subject Spinbox range.
        """
        if not ds_name: return

        try:
            # 1. Tell DataManager to instantiate this dataset
            self.data_manager.load_dataset_class(ds_name)
            
            # 2. Get Metadata from the instance
            meta = self.data_manager.get_metadata()
            
            # 3. Update Context Dropdown
            self.combo_context.clear()
            if meta and 'contexts' in meta:
                self.combo_context.addItems(meta['contexts'])
            
            # 4. Update Subject Spinbox Range
            if meta and 'subjects' in meta:
                sub_list = meta['subjects']
                if sub_list:
                    self.spin_sub.setRange(min(sub_list), max(sub_list))
                    self.spin_sub.setValue(min(sub_list))
            
            print(f"UI Updated for {ds_name}")

        except Exception as e:
            print(f"Error changing dataset to {ds_name}: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load dataset metadata: {e}")

    def load_data(self):
        ds_name = self.combo_dataset.currentText()
        sub_id = self.spin_sub.value()
        ctx = self.combo_context.currentText()

        self.lbl_status.setText("Loading data...")
        QApplication.processEvents() # Force UI update

        raw = self.data_manager.load_data(sub_id, ctx)
        if raw:
            self.session.load_data(raw, ds_name, sub_id, ctx)
            self.update_ui_after_change()
            self.lbl_status.setText(self.session.get_info_string())
        else:
            QMessageBox.critical(self, "Error", "Could not load data.")
            self.lbl_status.setText("Load failed.")

    def update_ui_after_change(self):
        """Refreshes the history list and the available processing steps."""
        # 1. Update History List
        self.list_history.clear()
        for step in self.session.history:
            # step is either a string (load) or a dict (transform)
            if isinstance(step, dict):
                self.list_history.addItem(f"{step.get('name')} {step.get('kwargs')}")
            else:
                self.list_history.addItem(str(step))

        # 2. Update Available Steps dropdown based on current data type (Raw/Epochs)
        avail_steps = Registry.get_available_steps(self.session.current_data)
        self.combo_steps.clear()
        # Store metadata in the item data
        for step in avail_steps:
            self.combo_steps.addItem(f"{step['name']} ({step['type']})", userData=step)

    def on_step_selected(self):
        """Regenerate the input form when the dropdown changes."""
        idx = self.combo_steps.currentIndex()
        if idx < 0: return
        
        step_data = self.combo_steps.currentData() # Get stored dict
        name = step_data['name']
        s_type = step_data['type']
        
        # Get defaults from Registry
        defaults = Registry.get_defaults(name, s_type)
        
        # Build form
        self.step_form.build_form(defaults)

    def apply_step(self):
        """Gather form data and execute transform."""
        if not self.session.current_data:
            return

        idx = self.combo_steps.currentIndex()
        step_data = self.combo_steps.currentData()
        
        # Get values from dynamic form
        kwargs = self.step_form.get_values()
        
        # Build Config
        step_config = {
            "type": step_data['type'],
            "name": step_data['name'],
            "module_name": "src.preprocessing.preprocessing_functions" if step_data['type'] == 'function' else None,
            "kwargs": kwargs
        }
        
        # Run
        success = self.session.run_transform(step_config)
        
        if success:
            self.update_ui_after_change()
            self.lbl_status.setText(self.session.get_info_string())
        else:
            QMessageBox.warning(self, "Error", "Transform failed. Check console.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BCIWorkstationGUI()
    window.show()
    sys.exit(app.exec())