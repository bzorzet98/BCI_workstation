import sys
import pprint
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGroupBox, QComboBox, 
                             QTextEdit, QHBoxLayout, QPushButton, QSizePolicy)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Use Qt backend
try:
    matplotlib.use('Qt5Agg')
except:
    pass

class InfoTab(QWidget):
    def __init__(self, session, data_manager):
        super().__init__()
        self.session = session
        self.data_manager = data_manager
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 1. Dataset General Info
        self.grp_general = QGroupBox("Dataset Information")
        self.lbl_general = QLabel("No dataset loaded.")
        self.lbl_general.setWordWrap(True)
        v1 = QVBoxLayout()
        v1.addWidget(self.lbl_general)
        self.grp_general.setLayout(v1)
        self.layout.addWidget(self.grp_general)
        
        # 2. Subject Extra Info (Scrollable JSON)
        self.grp_extra = QGroupBox("Subject Extra Info")
        self.txt_extra = QTextEdit()
        self.txt_extra.setReadOnly(True)
        self.txt_extra.setPlaceholderText("No extra info available.")
        v2 = QVBoxLayout()
        v2.addWidget(self.txt_extra)
        self.grp_extra.setLayout(v2)
        self.layout.addWidget(self.grp_extra)
        
        # 3. Demographics Chart
        self.grp_demo = QGroupBox("Subject Demographics")
        self.demo_layout = QVBoxLayout()
        self.grp_demo.setLayout(self.demo_layout)
        
        # Controls
        h_demo = QHBoxLayout()
        h_demo.addWidget(QLabel("Column:"))
        self.combo_demo_col = QComboBox()
        self.combo_demo_col.currentTextChanged.connect(self.plot_demographics)
        h_demo.addWidget(self.combo_demo_col)
        
        self.btn_plot = QPushButton("Refresh Plot")
        self.btn_plot.clicked.connect(self.plot_demographics)
        h_demo.addWidget(self.btn_plot)
        
        self.demo_layout.addLayout(h_demo)
        
        # Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200) 
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.demo_layout.addWidget(self.canvas)
        self.layout.addWidget(self.grp_demo)
        

    def refresh(self):
        """Called when new data is loaded."""
        ds = self.data_manager.active_dataset_instance
        
        # A. Update General Info
        if ds:
            txt = (f"Code: {getattr(ds, 'code', 'N/A')}\n"
                   f"Montage: {getattr(ds, 'standard_montage', 'N/A')}\n"
                   f"Sfreq: {getattr(ds, 'sfreq', 'N/A')} Hz\n"
                   f"Subjects: {len(ds.get_subjects())}\n"
                   f"Event IDs: {getattr(ds, 'event_id', 'N/A')}")
            self.lbl_general.setText(txt)
            
            # Update Demographics Dropdown
            # We call the method we added to BaseEEGDataset
            df = ds.get_subject_metadata() 
            
            self.combo_demo_col.blockSignals(True) # Prevent triggering plot during clear
            self.combo_demo_col.clear()
            if df is not None and not df.empty:
                self.combo_demo_col.addItems(df.columns.astype(str))
                self.combo_demo_col.setEnabled(True)
                self.btn_plot.setEnabled(True)
            else:
                self.combo_demo_col.addItem("No Metadata")
                self.combo_demo_col.setEnabled(False)
                self.btn_plot.setEnabled(False)
            self.combo_demo_col.blockSignals(False)
            
        else:
            self.lbl_general.setText("No dataset loaded.")
        
        # B. Update Extra Info (Subject Specific)
        if hasattr(self.session, 'extra_info') and self.session.extra_info:
            # Use pprint for nice dictionary formatting
            formatted_text = pprint.pformat(self.session.extra_info, indent=2)
            self.txt_extra.setText(formatted_text)
        else:
            self.txt_extra.clear()
        
        # C. Initial Plot
        self.plot_demographics()

    def plot_demographics(self):
        """Handles plotting Histograms or Pie Charts based on data type."""
        self.figure.clear()
        
        ds = self.data_manager.active_dataset_instance
        if not ds: return

        df = ds.get_subject_metadata()
        col_name = self.combo_demo_col.currentText()
        
        if df is None or col_name not in df.columns:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        data = df[col_name]
        
        try:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(data):
                # Histogram
                data.plot.hist(ax=ax, bins=10, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {col_name}")
                ax.set_xlabel(col_name)
                ax.set_ylabel("Count")
            else:
                # Pie Chart (Categorical)
                counts = data.value_counts()
                counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, cmap='Pastel1')
                ax.set_ylabel('') # Remove y-label
                ax.set_title(f"Distribution of {col_name}")
                
            self.figure.tight_layout()
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not plot: {e}", ha='center', va='center')
            
        self.canvas.draw()
        self.canvas.flush_events() # Ensures the GUI loop processes the paint event