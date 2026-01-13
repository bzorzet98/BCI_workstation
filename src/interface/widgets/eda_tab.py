import ast
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.core.registry import Registry
from src.utils.imports import import_class

class EDATab(QTabWidget):
    """
    Exploratory Data Analysis Tab.
    Separates Configuration (with History) and Results into top-level tabs.
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

        # Persistence
        self.inputs = {}
        self.history_data = [] 
        self.last_report = None

        self._init_ui()

    def _init_ui(self):
        # 1. Setup Configuration Tab (Contains History + Parameters)
        self.tab_config = QWidget()
        self._setup_config_tab()
        
        # 2. Setup Results Tab (Text + Visual Analytics)
        self.tab_results = QWidget()
        self._setup_results_tab()

        self.addTab(self.tab_config, "EDA Configuration")
        self.addTab(self.tab_results, "Diagnostic Results")
        
        # Trigger Initial Build
        self.on_analyzer_changed()

    def refresh(self):
        """Called by Main Window to sync state."""
        if sip.isdeleted(self): return
        self.on_analyzer_changed()
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    def _setup_config_tab(self):
        """Layout: [ History Sidebar | Parameter Setup ]"""
        layout = QHBoxLayout(self.tab_config)

        # LEFT: Internal History Sidebar
        hist_panel = QFrame()
        hist_panel.setFixedWidth(260)
        hist_panel.setStyleSheet(f"border-right: 1px solid {self.c_spine};")
        hist_vbox = QVBoxLayout(hist_panel)
        
        hist_vbox.addWidget(QLabel("<b>Analysis History:</b>"))
        self.list_history = QListWidget()
        self.list_history.setStyleSheet("background-color: white; color: black;")
        self.list_history.itemClicked.connect(self._load_from_history)
        hist_vbox.addWidget(self.list_history)
        
        self.btn_delete_hist = QPushButton("Remove Selected")
        self.btn_delete_hist.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        self.btn_delete_hist.clicked.connect(self._remove_selected_history)
        hist_vbox.addWidget(self.btn_delete_hist)
        
        layout.addWidget(hist_panel)

        # RIGHT: Parameter Setup
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
        params_layout.addWidget(QLabel("<b>Parameters:</b>"))
        self.scroll_config = QScrollArea()
        self.scroll_config.setWidgetResizable(True)
        self.scroll_config.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        self.form_widget = QWidget()
        self.form_widget.setStyleSheet("background-color: #f8f9fa;")
        self.form_layout = QFormLayout(self.form_widget)
        self.scroll_config.setWidget(self.form_widget)
        params_layout.addWidget(self.scroll_config)
        
        # Run Button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_layout.addWidget(self.btn_run)
        
        layout.addWidget(params_panel)

    def _setup_results_tab(self):
        """Layout: [ Result Navigation Tabs ]"""
        layout = QVBoxLayout(self.tab_results)
        self.inner_results = QTabWidget()
        
        # Result Tab 1: Text Summary
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet(f"background-color: {self.c_ax}; color: {self.c_text}; font-family: monospace;")
        
        # Result Tab 2: Visual Analytics
        plot_container = QWidget()
        p_vbox = QVBoxLayout(plot_container)
        
        # Selection Bar
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

        # Plot Controls (Log Scale, Z-Score, Z-MAD)
        h_opts = QHBoxLayout()
        self.chk_log_x = QCheckBox("Log Scale (X)")
        self.chk_zscore = QCheckBox("Z-Score")
        self.chk_zmad = QCheckBox("Z-MAD")
        
        self.chk_zscore.toggled.connect(lambda chk: self.chk_zmad.setChecked(False) if chk else None)
        self.chk_zmad.toggled.connect(lambda chk: self.chk_zscore.setChecked(False) if chk else None)

        for chk in [self.chk_log_x, self.chk_zscore, self.chk_zmad]:
            chk.stateChanged.connect(self.update_canvas)
            h_opts.addWidget(chk)
        h_opts.addStretch()
        p_vbox.addLayout(h_opts)
        
        # Matplotlib Canvas
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        p_vbox.addWidget(self.canvas)
        
        self.inner_results.addTab(self.txt_report, "Text Summary")
        self.inner_results.addTab(plot_container, "Visual Analytics")
        layout.addWidget(self.inner_results)

    # --- MATH & PLOTTING ---

    def _apply_transforms(self, data_list):
        data = np.array(data_list, dtype=float)
        if self.chk_zscore.isChecked():
            data = (data - np.mean(data)) / (np.std(data) + 1e-9)
        elif self.chk_zmad.isChecked():
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            data = (data - median) / (mad * 1.4826 + 1e-9)
        return data

    def update_canvas(self):
        if sip.isdeleted(self) or not self.last_report: return
        metric = self.cb_metric_selector.currentText()
        target_ch = self.cb_channel_selector.currentText()
        if not metric or not target_ch: return

        visuals_data = self.last_report['visuals'].get(metric, {})
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)

        # Labels
        x_label = metric
        if self.chk_zscore.isChecked(): x_label += " (Z-Score)"
        if self.chk_zmad.isChecked(): x_label += " (Z-MAD)"

        if target_ch == "All Channels (Overlay)":
            for ch, raw_data in visuals_data.items():
                transformed = self._apply_transforms(raw_data)
                ax.hist(transformed, bins=30, alpha=0.4, label=ch)
            ax.legend(fontsize=8, facecolor=self.c_ax, labelcolor=self.c_text)
        else:
            raw_data = visuals_data.get(target_ch, [])
            transformed = self._apply_transforms(raw_data)
            ax.hist(transformed, bins=30, color='#589df6', edgecolor='white', alpha=0.8)
            ax.set_title(f"{metric} - Channel: {target_ch}", color=self.c_text)

        ax.set_xlabel(x_label, color=self.c_text)
        ax.set_ylabel("Trial Count", color=self.c_text)
        if self.chk_log_x.isChecked(): ax.set_xscale('log', nonpositive='clip')
        
        ax.tick_params(colors=self.c_text)
        for spine in ax.spines.values(): spine.set_color(self.c_spine)
        self.canvas.draw()

    # --- LOGIC & EVENTS ---

    def on_analyzer_changed(self):
        if sip.isdeleted(self): return
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}
        
        name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
        params = Registry.get_analyzer_params(name, path)
        for arg, val in params.items():
            if arg in ['self', 'kwargs', 'args']: continue
            lbl = QLabel(arg)
            lbl.setStyleSheet("color: #333; font-weight: bold;")
            le = QLineEdit(str(val))
            le.setStyleSheet("background-color: white; color: black; border: 1px solid #ccc; padding: 4px;")
            self.form_layout.addRow(lbl, le)
            self.inputs[arg] = le

    def run_analysis(self):
        if not self.session.current_data: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            kwargs = {k: ast.literal_eval(v.text()) if v.text().strip() else "" for k, v in self.inputs.items()}
            name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            
            analyzer = Cls(**kwargs)
            report = analyzer.analyze(self.session.current_data)
            
            self.session.eda_report = report
            self.show_results(report)
            
            # Update History
            run_name = f"{name} - #{self.list_history.count() + 1}"
            self.history_data.append({'name': run_name, 'report': report})
            self.list_history.addItem(run_name)
            
            self.setCurrentIndex(1) # Jump to results tab
            main_win = self.window()
            if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()
                
        except Exception as e:
            self.txt_report.setText(f"Analysis Error: {e}")
            self.setCurrentIndex(1)
        finally:
            QApplication.restoreOverrideCursor()

    def show_results(self, report):
        if sip.isdeleted(self): return
        self.last_report = report
        self.txt_report.setText(report.get('summary', "Completed."))
        
        self.cb_metric_selector.blockSignals(True)
        self.cb_metric_selector.clear()
        visuals = report.get('visuals', {})
        if visuals: self.cb_metric_selector.addItems(list(visuals.keys()))
        self.cb_metric_selector.blockSignals(False)
        self._update_channel_options()

    def _update_channel_options(self):
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

    def _load_from_history(self, item):
        idx = self.list_history.row(item)
        if idx < len(self.history_data):
            report = self.history_data[idx]['report']
            self.session.eda_report = report
            self.show_results(report)
            self.setCurrentIndex(1) # Jump to results tab
            main_win = self.window()
            if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0:
            self.history_data.pop(row)
            self.list_history.takeItem(row)
            self.session.eda_report = None
            main_win = self.window()
            if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()