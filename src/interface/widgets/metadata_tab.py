import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGroupBox, QComboBox, 
                             QHBoxLayout, QPushButton, QSizePolicy)

class SubjectMetadataTab(QWidget):
    def __init__(self, session, data_manager, style_kwargs=None):
        super().__init__()
        self.session = session
        self.data_manager = data_manager
        
        # Style
        self.style_kwargs = style_kwargs or {}
        styles = self.style_kwargs
        self.c_bg = styles.get("figure.facecolor", "#2b2d30")
        self.c_ax = styles.get("axes.facecolor", "#1e1f22")
        self.c_text = styles.get("text.color", "#cfd8dc")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Controls
        h_ctrl = QHBoxLayout()
        h_ctrl.addWidget(QLabel("Metadata Column:"))
        self.combo_col = QComboBox()
        self.combo_col.currentTextChanged.connect(self.update_plot)
        h_ctrl.addWidget(self.combo_col)
        h_ctrl.addStretch()
        self.layout.addLayout(h_ctrl)
        
        # Plot Area
        self.figure = Figure(facecolor=self.c_bg)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas)

    def refresh(self):
        ds = self.data_manager.active_dataset_instance
        self.combo_col.blockSignals(True)
        self.combo_col.clear()
        
        if ds:
            df = ds.get_subject_metadata()
            if df is not None and not df.empty:
                self.combo_col.addItems(df.columns.astype(str))
            else:
                self.combo_col.addItem("No Metadata CSV Found")
        
        self.combo_col.blockSignals(False)
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        
        ds = self.data_manager.active_dataset_instance
        if not ds: return
        
        df = ds.get_subject_metadata()
        col = self.combo_col.currentText()
        
        if df is None or col not in df.columns:
            self.canvas.draw()
            return

        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)
        
        # Style Axes
        ax.spines['bottom'].set_color(self.c_text)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.c_text)
        ax.tick_params(colors=self.c_text)
        
        data = df[col]
        
        try:
            if pd.api.types.is_numeric_dtype(data):
                # Histogram
                data.plot.hist(ax=ax, bins=10, alpha=0.7)
                ax.set_title(f"Distribution of {col}", color=self.c_text)
            else:
                # Pie Chart
                counts = data.value_counts()
                counts.plot.pie(ax=ax, autopct='%1.1f%%', 
                                startangle=90, 
                                textprops={'color':self.c_text}) # Pie text always white
                ax.set_ylabel('')
                ax.set_title(f"Distribution of {col}", color=self.c_text)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", color='red', ha='center')
            
        self.canvas.draw()