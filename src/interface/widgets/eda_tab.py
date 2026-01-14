import ast
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox,QSpinBox)
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
        self.event_colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c', '#9b59b6']

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
        layout = QVBoxLayout(self.tab_results)
        self.res_subtabs = QTabWidget()
        
        # 1. Summary
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        
        # 2. Histogram (Big Picture)
        self.tab_hist = self._create_plot_subtab(self.update_hist_canvas, mode="hist")
        
        # 3. Scatter Plot (Feature Correlation)
        self.tab_scatter = self._create_plot_subtab(self.update_scatter_canvas, mode="scatter")
        
        # 4. Trial Mapping (Cue Analytics)
        self.tab_mapping = self._create_plot_subtab(self.update_mapping_canvas, mode="mapping")
        
        # 5. Signal Inspection (Ground Truth Microscope)
        self.tab_inspect = self._create_plot_subtab(self.update_inspect_canvas, mode="inspect")

        self.res_subtabs.addTab(self.txt_report, "Text Summary")
        self.res_subtabs.addTab(self.tab_hist, "Histograms")
        self.res_subtabs.addTab(self.tab_scatter, "Feature Scatter")
        self.res_subtabs.addTab(self.tab_mapping, "Trial Mapping")
        self.res_subtabs.addTab(self.tab_inspect, "Signal Inspection")
        layout.addWidget(self.res_subtabs)


    def _create_plot_subtab(self, callback, mode="hist"):
        w = QWidget(); lay = QVBoxLayout(w); h_ctrl = QHBoxLayout()
        w.cb_ch = QComboBox(); h_ctrl.addWidget(QLabel("Channel:")); h_ctrl.addWidget(w.cb_ch)
        w.cb_ch.currentIndexChanged.connect(callback)

        if mode in ["hist", "mapping"]:
            w.cb_metric = QComboBox(); h_ctrl.addWidget(QLabel("Metric:")); h_ctrl.addWidget(w.cb_metric)
            w.cb_metric.currentIndexChanged.connect(callback)
            
        if mode == "scatter":
            w.cb_metric_x = QComboBox(); w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X Metric:")); h_ctrl.addWidget(w.cb_metric_x)
            h_ctrl.addWidget(QLabel("Y Metric:")); h_ctrl.addWidget(w.cb_metric_y)
            w.cb_metric_x.currentIndexChanged.connect(callback); w.cb_metric_y.currentIndexChanged.connect(callback)

        if mode == "inspect":
            h_ctrl.addWidget(QLabel("Trial ID:")); w.sb_trial = QSpinBox()
            w.sb_trial.valueChanged.connect(callback); h_ctrl.addWidget(w.sb_trial)

        if mode != "inspect": # Math options for analytics tabs
            w.chk_log = QCheckBox("Log"); w.chk_z = QCheckBox("Z-Score"); w.chk_mad = QCheckBox("Z-MAD")
            for c in [w.chk_log, w.chk_z, w.chk_mad]: 
                c.stateChanged.connect(callback); h_ctrl.addWidget(c)

        h_ctrl.addStretch(); lay.addLayout(h_ctrl)
        w.fig = Figure(facecolor=self.c_bg, tight_layout=True); w.canvas = FigureCanvas(w.fig)
        lay.addWidget(w.canvas); return w

    
    def _sync_selectors(self):
        """Ensures all plot tabs are looking at the same Metric/Channel."""
        if sip.isdeleted(self) or not self.last_report: return
        
        # Get values from the sender (the combobox that was clicked)
        sender = self.sender()
        if not isinstance(sender, QComboBox): return
        
        val = sender.currentText()
        is_metric = "cb_metric" in sender.objectName() # Optional: if you name them
        
        # For simplicity, we just update all canvases when a metric changes
        self.update_hist_canvas()
        self.update_scatter_canvas()
        self.update_detail_canvas()  
        
    def _style_ax(self, ax):
        """Standardizes axis appearance."""
        ax.set_facecolor(self.c_ax)
        ax.tick_params(colors=self.c_text)
        for s in ax.spines.values(): s.set_color(self.c_spine)
        ax.xaxis.label.set_color(self.c_text)
        ax.yaxis.label.set_color(self.c_text)
        ax.title.set_color(self.c_text)
        
    def update_hist_canvas(self):
        if not self.last_report: return
        w = self.tab_hist; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return
        
        data_dict = self.last_report['metrics'][metric]['values']
        if ch == "All Channels (Overlay)":
            for name, raw in data_dict.items():
                ax.hist(self._apply_transforms(raw, w), bins=30, alpha=0.4, label=name)
            ax.legend(fontsize=8)
        else:
            ax.hist(self._apply_transforms(data_dict[ch], w), bins=30, color='#589df6', edgecolor='white', alpha=0.8)
        w.canvas.draw()

    def update_scatter_canvas(self):
        if not self.last_report: return
        w = self.tab_scatter; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        if not mx or not my or not ch or ch == "All Channels (Overlay)": return

        x_vals = self._apply_transforms(self.last_report['metrics'][mx]['values'][ch], w)
        y_vals = self._apply_transforms(self.last_report['metrics'][my]['values'][ch], w)
        bad_idx = self.last_report['artifacts']['bad_indices']
        
        ax.scatter(x_vals, y_vals, color='gray', alpha=0.2, label="Normal")
        ax.scatter(x_vals[bad_idx], y_vals[bad_idx], color='red', marker='x', s=80, label="Flagged")
        ax.set_xlabel(mx); ax.set_ylabel(my); ax.legend(); w.canvas.draw()

    def update_mapping_canvas(self):
        if not self.last_report: return
        w = self.tab_mapping; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch or ch == "All Channels (Overlay)": return

        y_vals = self._apply_transforms(self.last_report['metrics'][metric]['values'][ch], w)
        events = np.array(self.last_report['artifacts']['event_types'])
        bad_idx = self.last_report['artifacts']['bad_indices']
        
        for i, etype in enumerate(np.unique(events)):
            mask = (events == etype)
            ax.scatter(np.where(mask)[0], y_vals[mask], color=self.event_colors[i % 5], label=f"Cue {etype}", alpha=0.6)

        ax.scatter(bad_idx, y_vals[bad_idx], color='red', marker='x', s=100, label="Flagged")
        ax.set_title(f"Trial Index vs {metric}"); ax.legend(); w.canvas.draw()

    def update_inspect_canvas(self):
        if not self.last_report or 'inspection_data' not in self.last_report: return
        w = self.tab_inspect; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        ch, tid = w.cb_ch.currentText(), w.sb_trial.value()
        if not ch or ch == "All Channels (Overlay)": return

        raw_trace = self.last_report['inspection_data']['raw_traces'][ch][tid]
        env_trace = self.last_report['inspection_data']['envelope_traces'][ch][tid]
        thresh = self.last_report['threshold_lines'][ch]
        t = np.linspace(self.last_report['artifacts']['tmin'], self.last_report['artifacts']['tmax'], len(raw_trace))

        ax.plot(t, np.array(raw_trace)*1e6, color='gray', alpha=0.3, label="Raw Filtered (uV)")
        ax.plot(t, np.array(env_trace)*1e6, color='#589df6', lw=2, label="Envelope")
        ax.axhline(y=thresh*1e6, color='red', ls='--', label="Threshold")
        ax.set_title(f"Ground Truth Inspection - Trial #{tid}"); ax.legend(); w.canvas.draw()

    # --- MATH & PLOTTING ---
    def show_results(self, report):
        self.last_report = report
        self.txt_report.setText(report.get('summary', ""))
        
        metrics_list = list(report['metrics'].keys())
        chans_list = list(report['metrics'][metrics_list[0]]['values'].keys())
        n_trials = len(report['artifacts']['event_types'])

        for tab in [self.tab_hist, self.tab_scatter, self.tab_mapping, self.tab_inspect]:
            tab.cb_ch.blockSignals(True); tab.cb_ch.clear(); tab.cb_ch.addItem("All Channels (Overlay)"); tab.cb_ch.addItems(chans_list); tab.cb_ch.blockSignals(False)
            
            if hasattr(tab, 'cb_metric'):
                tab.cb_metric.blockSignals(True); tab.cb_metric.clear(); tab.cb_metric.addItems(metrics_list); tab.cb_metric.blockSignals(False)
            if hasattr(tab, 'cb_metric_x'):
                for cb in [tab.cb_metric_x, tab.cb_metric_y]:
                    cb.blockSignals(True); cb.clear(); cb.addItems(metrics_list); cb.blockSignals(False)
            if hasattr(tab, 'sb_trial'):
                tab.sb_trial.setRange(0, n_trials - 1)

        self.update_hist_canvas(); self.update_scatter_canvas(); 
        self.update_mapping_canvas(); self.update_inspect_canvas()

    def on_analyzer_changed(self):
        if sip.isdeleted(self): return
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.inputs = {}
        name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
        params = Registry.get_analyzer_params(name, path)
        for arg, val in params.items():
            if arg in ['self', 'kwargs']: continue
            lbl = QLabel(arg); le = QLineEdit(str(val))
            self.form_layout.addRow(lbl, le); self.inputs[arg] = le

    def run_analysis(self):
        if not self.session.current_data: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            kwargs = {k: ast.literal_eval(v.text()) if v.text().strip() else "" for k, v in self.inputs.items()}
            Cls = import_class(self.combo_analyzer.currentText(), self.combo_analyzer.currentData())
            
            analyzer = Cls( **kwargs)
            report = analyzer.analyze(self.session.current_data)
            self.session.eda_report = report
            
            # Save history
            name = self.combo_analyzer.currentText()
            self.history_data.append({'name': f"{name} #{self.list_history.count()+1}", 'report': report})
            self.list_history.addItem(self.history_data[-1]['name'])
            
            self.show_results(report); self.setCurrentIndex(1)
            if hasattr(self.window(), 'refresh_gui'): self.window().refresh_gui()
        except Exception as e: self.txt_report.setText(f"Error: {e}"); self.setCurrentIndex(1)
        finally: QApplication.restoreOverrideCursor()

    def _load_from_history(self, item):
        idx = self.list_history.row(item)
        report = self.history_data[idx]['report']
        self.session.eda_report = report
        self.show_results(report)
        self.setCurrentIndex(1) # Jump to results tab
        main_win = self.window()
        if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0:
            self.history_data.pop(row); self.list_history.takeItem(row)
            self.session.eda_report = None
            main_win = self.window()
            if hasattr(main_win, 'refresh_gui'): main_win.refresh_gui()
            
    def _apply_transforms(self, data_input, sub_tab):
        """
        Applies Log, Z-Score, or Z-MAD to data based on UI checkboxes.
        Handles both individual thresholds and full data arrays.
        """
        data = np.array(data_input, dtype=float)
        
        # 1. Log Transformation (applied first to compress dynamic range)
        if hasattr(sub_tab, 'chk_log') and sub_tab.chk_log.isChecked():
            data = np.log10(np.maximum(data, 1e-12))
            
        # 2. Normalization (Z-Score takes priority over Z-MAD)
        if hasattr(sub_tab, 'chk_z') and sub_tab.chk_z.isChecked():
            # Use global stats from the whole channel session to normalize
            mu, sigma = np.mean(data), np.std(data)
            data = (data - mu) / (sigma + 1e-9)
            
        elif hasattr(sub_tab, 'chk_mad') and sub_tab.chk_mad.isChecked():
            # Robust normalization using Median Absolute Deviation
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            data = (data - median) / (mad * 1.4826 + 1e-9)
            
        return data