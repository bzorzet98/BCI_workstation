import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QHBoxLayout, QSizePolicy, QTabWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt

class DatasetMetadata(QWidget):
    def __init__(self, session, data_manager, style_kwargs=None):
        super().__init__()
        self.session = session
        self.data_manager = data_manager
        
        # --- STYLE PARSING ---
        self.style_kwargs = style_kwargs or {}
        styles = self.style_kwargs
        self.c_bg = styles.get("figure.facecolor", "#2b2d30")
        self.c_ax = styles.get("axes.facecolor", "#1e1f22")
        self.c_text = styles.get("text.color", "#cfd8dc")
        self.c_spine = styles.get("axes.edgecolor", "#4e5154")

        self._init_ui()

    def _init_ui(self):
        """Builds the main layout with internal tabs."""
        self.main_layout = QVBoxLayout(self)
        
        # Internal Tab Widget to separate Plots and Data
        self.sub_tabs = QTabWidget()
        
        # 1. Visualization Sub-Tab
        self.tab_viz = QWidget()
        self._setup_visualization_tab()
        
        # 2. Raw Data Sub-Tab
        self.tab_table = QWidget()
        self._setup_table_tab()
        
        self.sub_tabs.addTab(self.tab_viz, "Visual Analytics")
        self.sub_tabs.addTab(self.tab_table, "Metadata Table")
        
        self.main_layout.addWidget(self.sub_tabs)

    def _setup_visualization_tab(self):
        layout = QVBoxLayout(self.tab_viz)
        
        # Controls Bar
        h_ctrl = QHBoxLayout()
        h_ctrl.addWidget(QLabel("<b>Metadata Column to Analyze:</b>"))
        self.combo_col = QComboBox()
        self.combo_col.currentTextChanged.connect(self.update_plot)
        h_ctrl.addWidget(self.combo_col)
        h_ctrl.addStretch()
        layout.addLayout(h_ctrl)
        
        # Plotting Area
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        
        layout.addWidget(self.canvas)

    def _setup_table_tab(self):
        layout = QVBoxLayout(self.tab_table)
        
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        # Clean styling for the table
        self.table.setStyleSheet("""
            QTableWidget { background-color: #f8f9fa; color: #212529; gridline-color: #dee2e6; }
            QHeaderView::section { background-color: #e9ecef; padding: 4px; border: 1px solid #dee2e6; font-weight: bold; }
        """)
        
        layout.addWidget(self.table)

    # --- PUBLIC API ---

    def refresh(self):
        """Loads data from the manager and populates both the dropdown and the table."""
        ds = self.data_manager.active_dataset_instance
        if not ds: return

        df = ds.get_subject_metadata()
        
        # 1. Update Column Dropdown
        self.combo_col.blockSignals(True)
        self.combo_col.clear()
        
        if df is not None and not df.empty:
            self.combo_col.addItems(df.columns.astype(str))
            # 2. Populate Raw Table
            self._populate_table(df)
        else:
            self.combo_col.addItem("No Metadata Found")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        
        self.combo_col.blockSignals(False)
        self.update_plot()

    def update_plot(self):
        """Updates the Matplotlib chart based on the selected column."""
        self.ax.clear()
        self._setup_ax_style()
        
        ds = self.data_manager.active_dataset_instance
        df = ds.get_subject_metadata() if ds else None
        col = self.combo_col.currentText()
        
        if df is None or col not in df.columns or df.empty:
            self.canvas.draw()
            return

        data = df[col]
        try:
            if pd.api.types.is_numeric_dtype(data):
                # Histogram for Numbers
                data.plot.hist(ax=self.ax, bins=15, color='#2196F3', alpha=0.7, edgecolor=self.c_spine)
                self.ax.set_ylabel("Subject Count", color=self.c_text)
            else:
                # Pie Chart for Categories
                counts = data.value_counts()
                counts.plot.pie(ax=self.ax, autopct='%1.1f%%', startangle=90, 
                                textprops={'color': self.c_text, 'weight': 'bold'})
                self.ax.set_ylabel('') # Remove default 'count' label
            
            self.ax.set_title(f"Distribution of {col}", color=self.c_text, fontweight='bold')
            
        except Exception as e:
            self.ax.text(0.5, 0.5, f"Plotting Error:\n{str(e)}", color='red', ha='center', va='center', transform=self.ax.transAxes)
            
        self.canvas.draw()

    # --- PRIVATE HELPERS ---

    def _setup_ax_style(self):
        self.ax.set_facecolor(self.c_ax)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['left'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(colors=self.c_text)

    def _populate_table(self, df):
        """Converts the pandas DataFrame into the QTableWidget."""
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns.astype(str))
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iloc[i, j]
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable) # Read-only
                self.table.setItem(i, j, item)
        
        # Resize columns to fit content
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)