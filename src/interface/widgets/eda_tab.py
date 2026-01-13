import ast
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.core.registry import Registry
from src.utils.imports import import_class

class EDATab(QTabWidget):
    """
    Exploratory Data Analysis Tab.
    Provides a configuration panel for custom analyzers and a results panel
    with automated reporting and visualization.
    """
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # --- AESTHETICS ---
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")

        # Data Persistence
        self.inputs = {}
        self.history_data = [] 
        self.last_report = None

        self._init_ui()

    def _init_ui(self):
        """Initializes the top-level tab structure."""
        # 1. Configuration Tab
        self.tab_config = QWidget()
        self._setup_config_layout()
        
        # 2. Results Tab
        self.tab_results = QWidget()
        self._setup_results_layout()

        self.addTab(self.tab_config, "EDA Configuration")
        self.addTab(self.tab_results, "Diagnostic Results")
        
        # Initial population
        self.on_analyzer_changed()

    def _setup_config_layout(self):
        """Builds the History sidebar and the Dynamic Parameter form."""
        layout = QHBoxLayout(self.tab_config)
        
        # LEFT: History Panel
        hist_panel = QFrame()
        hist_panel.setFixedWidth(250)
        hist_vbox = QVBoxLayout(hist_panel)
        hist_vbox.addWidget(QLabel("<b>Analysis History:</b>"))
        
        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self._load_from_history)
        hist_vbox.addWidget(self.list_history)
        layout.addWidget(hist_panel)
        
        # RIGHT: Analyzer Setup
        params_panel = QWidget()
        params_layout = QVBoxLayout(params_panel)
        
        # Analyzer Selector
        f_select = QFormLayout()
        self.combo_analyzer = QComboBox()
        available = Registry.discover_eda_analyzers()
        for name, path in available.items():
            self.combo_analyzer.addItem(name, userData=path)
        
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)
        f_select.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)
        params_layout.addLayout(f_select)

        # Dynamic Kwargs Scroll Area
        params_layout.addWidget(QLabel("<b>Parameter Configuration:</b>"))
        self.scroll_config = QScrollArea()
        self.scroll_config.setWidgetResizable(True)
        self.scroll_config.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        self.form_widget = QWidget()
        self.form_widget.setStyleSheet("background-color: #f8f9fa;")
        self.form_layout = QFormLayout(self.form_widget)
        self.scroll_config.setWidget(self.form_widget)
        params_layout.addWidget(self.scroll_config)
        
        # Action Button
        self.btn_run = QPushButton("Run Analysis & View Results")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_layout.addWidget(self.btn_run)
        
        layout.addWidget(params_panel)

    def _setup_results_layout(self):
        """Builds the internal tabs for Text Summary and Visual Analytics."""
        layout = QVBoxLayout(self.tab_results)
        self.inner_results = QTabWidget()
        
        # 1. Text Summary
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet(f"background-color: {self.c_ax}; color: {self.c_text}; font-family: monospace;")
        
        # 2. Visual Analytics
        plot_container = QWidget()
        p_vbox = QVBoxLayout(plot_container)
        
        # Selectors Bar
        h_ctrl = QHBoxLayout()
        self.cb_metric_selector = QComboBox()
        self.cb_channel_selector = QComboBox() 
        self.cb_metric_selector.currentIndexChanged.connect(self._update_channel_options)
        self.cb_channel_selector.currentIndexChanged.connect(self.update_canvas)

        h_ctrl.addWidget(QLabel("Metric:"))
        h_ctrl.addWidget(self.cb_metric_selector)
        h_ctrl.addWidget(QLabel("Channel:"))
        h_ctrl.addWidget(self.cb_channel_selector)
        h_ctrl.addStretch()
        p_vbox.addLayout(h_ctrl)
        
        # Matplotlib Canvas
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        p_vbox.addWidget(self.canvas)
        
        self.inner_results.addTab(self.txt_report, "Text Summary")
        self.inner_results.addTab(plot_container, "Visual Analytics")
        layout.addWidget(self.inner_results)

    # --- LOGIC & EVENTS ---

    def on_analyzer_changed(self):
        """Rebuilds the parameter form when a new analyzer is selected."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}

        name = self.combo_analyzer.currentText()
        path = self.combo_analyzer.currentData()
        params = Registry.get_analyzer_params(name, path)

        for arg, val in params.items():
            if arg in ['picks', 'self', 'kwargs', 'args']: continue
            
            lbl = QLabel(arg)
            lbl.setStyleSheet("color: #333; font-weight: bold;")
            le = QLineEdit(str(val))
            le.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc; padding: 4px;")
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def run_analysis(self):
        """Instantiates and executes the selected analyzer on current session data."""
        if not self.session.current_data:
            self.txt_report.setText("Error: No data loaded in session.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # 1. Parse Arguments
            kwargs = {}
            for arg, le in self.inputs.items():
                try: kwargs[arg] = ast.literal_eval(le.text())
                except: kwargs[arg] = le.text()

            # 2. Instantiate and Analyze
            name = self.combo_analyzer.currentText()
            path = self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            
            kwargs.pop('picks', None)
            analyzer = Cls(picks=self.session.selected_channels, **kwargs)
            report = analyzer.analyze(self.session.current_data)
            
            # 3. Save results
            self.session.eda_report = report
            run_name = f"{name} - Run #{self.list_history.count() + 1}"
            self.history_data.append({'name': run_name, 'report': report})
            self.list_history.addItem(run_name)
            
            # 4. Update UI
            self.show_results(report)
            self.setCurrentIndex(1) # Jump to results tab
            
            # 5. Global Refresh
            main_win = self.window()
            if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()
                
        except Exception as e:
            self.txt_report.setText(f"Analysis Error ({name}):\n{str(e)}")
            self.setCurrentIndex(1)
            
        finally:
            QApplication.restoreOverrideCursor()

    def show_results(self, report):
        """Populates text and plot selectors with report data."""
        self.last_report = report
        self.txt_report.setText(report.get('summary', "No summary provided."))

        # Update Metric Selector
        self.cb_metric_selector.blockSignals(True)
        self.cb_metric_selector.clear()
        visuals = report.get('visuals', {})
        if visuals:
            self.cb_metric_selector.addItems(list(visuals.keys()))
        self.cb_metric_selector.blockSignals(False)

        self._update_channel_options()

    def _update_channel_options(self):
        """Updates channel list based on the currently selected metric."""
        if not self.last_report: return
        metric = self.cb_metric_selector.currentText()
        if not metric: return

        channels = list(self.last_report['visuals'].get(metric, {}).keys())
        
        self.cb_channel_selector.blockSignals(True)
        self.cb_channel_selector.clear()
        self.cb_channel_selector.addItem("All Channels (Overlay)")
        self.cb_channel_selector.addItems(channels)
        self.cb_channel_selector.blockSignals(False)
        self.update_canvas()

    def update_canvas(self):
        """Renders the selected metric/channel histogram."""
        if not self.last_report: return
        
        metric = self.cb_metric_selector.currentText()
        target_ch = self.cb_channel_selector.currentText()
        if not metric or not target_ch: return

        visuals_data = self.last_report['visuals'].get(metric, {})

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)

        if target_ch == "All Channels (Overlay)":
            for ch, data in visuals_data.items():
                ax.hist(data, bins=30, alpha=0.4, label=ch)
            ax.legend(fontsize=8, facecolor=self.c_ax, labelcolor=self.c_text)
            ax.set_title(f"Global Distribution: {metric}", color=self.c_text)
        else:
            data = visuals_data.get(target_ch, [])
            ax.hist(data, bins=30, color='#589df6', edgecolor='white', alpha=0.8)
            ax.set_title(f"{metric} - Channel: {target_ch}", color=self.c_text)

        # Apply Styling
        ax.tick_params(colors=self.c_text)
        for spine in ax.spines.values(): spine.set_color(self.c_spine)
        self.canvas.draw()

    def _load_from_history(self, item):
        """Retrieves and displays a previous run from the history list."""
        idx = self.list_history.row(item)
        if idx < len(self.history_data):
            self.show_results(self.history_data[idx]['report'])
            self.setCurrentIndex(1)