import ast
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox,QSpinBox,QGroupBox,QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
import inspect
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mne
from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox

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
        
        # 1. Sync Focus Signal list
        self.combo_focus.blockSignals(True)
        self.combo_focus.clear()
        # Add all available signals in workspace
        self.combo_focus.addItems(list(self.session.workspace.keys()))
        # Set current selection to session's active context
        if self.session.active_context:
            idx = self.combo_focus.findText(self.session.active_context)
            if idx >= 0: self.combo_focus.setCurrentIndex(idx)
        self.combo_focus.blockSignals(False)

        # 2. Populate Reference signals
        self._populate_ref_signals()

        # 3. Sync Analyzer Logic
        self.on_analyzer_changed()
        
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

        if self.session.current_data:
            self._populate_ref_signals()
            self.on_analyzer_changed()

        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    #     # RIGHT: Parameter Setup
    #     params_panel = QWidget()
    #     params_layout = QVBoxLayout(params_panel)
        
    #     # Analyzer Selector
    #     f_select = QFormLayout()
    #     self.combo_analyzer = QComboBox()
    #     available = Registry.discover_eda_analyzers()
    #     for name, path in available.items():
    #         self.combo_analyzer.addItem(name, userData=path)
        
    #     self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)
    #     f_select.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)
    #     params_layout.addLayout(f_select)

    #     self.ref_signal_container = QGroupBox("Reference / Context Signals")
    #     self.ref_signal_container.setCheckable(False) # The group itself isn't a toggle
    #     ref_layout = QVBoxLayout(self.ref_signal_container)

    #     self.list_ref_signals = QListWidget()
    #     self.list_ref_signals.setFixedHeight(120) # Keep it compact
    #     ref_layout.addWidget(QLabel("Select signals to use as reference:"))
    #     ref_layout.addWidget(self.list_ref_signals)

    #     params_layout.addWidget(self.ref_signal_container)
    #     self.ref_signal_container.hide() # Hidden by default

    #     # Dynamic Kwargs Scroll Area
    #     params_layout.addWidget(QLabel("<b>Parameters:</b>"))
    #     self.scroll_config = QScrollArea()
    #     self.scroll_config.setWidgetResizable(True)
    #     self.scroll_config.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
    #     self.form_widget = QWidget()
    #     self.form_widget.setStyleSheet("background-color: #f8f9fa;")
    #     self.form_layout = QFormLayout(self.form_widget)
    #     self.scroll_config.setWidget(self.form_widget)
    #     params_layout.addWidget(self.scroll_config)
        
    #     # Run Button
    #     self.btn_run = QPushButton("Run Analysis")
    #     self.btn_run.setFixedHeight(45)
    #     self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
    #     self.btn_run.clicked.connect(self.run_analysis)
    #     params_layout.addWidget(self.btn_run)
        
    #     layout.addWidget(params_panel)

    def _setup_config_tab(self):
        """Layout: [ History Sidebar | Parameter Setup ]"""
        layout = QHBoxLayout(self.tab_config)

        # LEFT: Internal History Sidebar (Keep this as you have it)

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
        
        # --- TOP FORM SELECTORS ---
        f_select = QFormLayout()
        
         # 1. NEW: Focus Signal Row (Matches your other tabs)
        self.combo_focus = QComboBox()
        self.combo_focus.currentIndexChanged.connect(self._on_focus_changed)
        f_select.addRow("<b>Focus Signal:</b>", self.combo_focus)

        # 1. Target Analyzer Row
        self.combo_analyzer = QComboBox()
        available = Registry.discover_eda_analyzers()
        for name, path in available.items():
            self.combo_analyzer.addItem(name, userData=path)
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)
        f_select.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)

        # 2. Compact Reference Selector Row
        self.lbl_ref_title = QLabel("<b>Reference Signal(s):</b>")
        self.combo_ref_signals = CheckableComboBox()
        self.combo_ref_signals.setEditable(True)
        self.combo_ref_signals.lineEdit().setReadOnly(True)
        
        f_select.addRow(self.lbl_ref_title, self.combo_ref_signals)
        params_layout.addLayout(f_select)

        # --- PARAMETERS SCROLL AREA ---
        params_layout.addWidget(QLabel("<b>Parameters:</b>"))
        self.scroll_config = QScrollArea()
        self.scroll_config.setWidgetResizable(True)
        self.scroll_config.setStyleSheet(f"background-color: {self.c_bg}; border: 1px solid {self.c_spine};")
        
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        self.scroll_config.setWidget(self.form_widget)
        params_layout.addWidget(self.scroll_config)
        
        # --- RUN BUTTON ---
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_layout.addWidget(self.btn_run)
        
        layout.addWidget(params_panel)

    def _on_focus_changed(self):
        """Updates the session focus when the user changes it in this tab."""
        new_focus = self.combo_focus.currentText()
        if new_focus:
            self.session.active_context = new_focus
            print(f"[EDA Tab] Focus shifted to: {new_focus}")

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
        
    def _is_trial_flagged_for_context(self, trial_idx, channel):
        """
        Returns True if the trial was flagged specifically because of this channel.
        """
        if not self.last_report: return False
        
        bad_indices = self.last_report['artifacts']['bad_indices']
        if trial_idx not in bad_indices: return False
        
        # Find the tags for this specific trial
        pos = bad_indices.index(trial_idx)
        tags = self.last_report['artifacts']['bad_tags'][pos] # List of strings
        
        # We only care if THIS channel appears in the tags
        search_term = channel.lower()
        for t in tags:
            if search_term in t.lower():
                return True
        return False

    def update_scatter_canvas(self):
        """Feature Correlation: X-Metric vs Y-Metric."""
        if not self.last_report: return
        w = self.tab_scatter; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        
        if not mx or not my or not ch or ch == "All Channels (Overlay)": 
            ax.text(0.5, 0.5, "Select a channel", ha='center', color='gray'); w.canvas.draw(); return

        # 1. Get Values
        raw_x = self.last_report['metrics'][mx]['values'][ch]
        raw_y = self.last_report['metrics'][my]['values'][ch]
        
        x_vals = self._apply_transforms(raw_x, w)
        y_vals = self._apply_transforms(raw_y, w)
        events = np.array(self.last_report['artifacts'].get('event_types', []))

        # 2. Plot Points by Cue Type
        unique_cues = np.unique(events)
        for i, cue_id in enumerate(unique_cues):
            mask = (events == cue_id)
            ax.scatter(x_vals[mask], y_vals[mask], 
                       color=self.event_colors[i % len(self.event_colors)], 
                       label=f"Cue {cue_id}", alpha=0.4, s=50)

        # 3. --- FIX: SMART FLAGGING ---
        # We must plot the Red X at the (Metric_X, Metric_Y) coordinate, not the index!
        flagged_x, flagged_y = [], []
        for i in range(len(y_vals)):
            if self._is_trial_flagged_for_context(i, ch):
                flagged_x.append(x_vals[i]) # Value of X-Metric for this trial
                flagged_y.append(y_vals[i]) # Value of Y-Metric for this trial
        
        if flagged_x:
            ax.scatter(flagged_x, flagged_y, color='red', marker='x', 
                       s=120, linewidth=2, label=f"Flagged {ch}", zorder=5)

        ax.set_xlabel(mx); ax.set_ylabel(my); ax.legend(); w.canvas.draw()

    def update_mapping_canvas(self):
        """Temporal Mapping: Trial Index vs Metric."""
        if not self.last_report: return
        w = self.tab_mapping; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        
        if not metric or not ch or ch == "All Channels (Overlay)": 
            ax.text(0.5, 0.5, "Select a channel", ha='center', color='gray'); w.canvas.draw(); return

        raw_y = self.last_report['metrics'][metric]['values'][ch]
        y_vals = self._apply_transforms(raw_y, w)
        x_indices = np.arange(len(y_vals))
        events = np.array(self.last_report['artifacts']['event_types'])
        
        # 1. Plot Cues
        unique_cues = np.unique(events)
        for i, cue_id in enumerate(unique_cues):
            mask = (events == cue_id)
            ax.scatter(x_indices[mask], y_vals[mask], 
                       color=self.event_colors[i % len(self.event_colors)], 
                       label=f"Cue {cue_id}", alpha=0.4, s=50)

        # 2. --- SMART FLAGGING ---
        # Here X IS the index, so it was working, but now it's channel-aware
        flagged_idx, flagged_y = [], []
        for i in range(len(y_vals)):
            if self._is_trial_flagged_for_context(i, ch):
                flagged_idx.append(i)
                flagged_y.append(y_vals[i])
        
        if flagged_idx:
            ax.scatter(flagged_idx, flagged_y, color='red', marker='x', 
                       s=120, linewidth=2, label=f"Flagged {ch}", zorder=5)

        ax.set_title(f"Trial Index vs {metric} ({ch})")
        ax.set_xlabel("Trial Index"); ax.legend(); w.canvas.draw()
        
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

    # def on_analyzer_changed(self):
    #     if sip.isdeleted(self): return
    #     while self.form_layout.count():
    #         item = self.form_layout.takeAt(0)
    #         if item.widget(): item.widget().deleteLater()
    #     self.inputs = {}
    #     name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
    #     Cls = import_class(name, path)
        
    #     # Inspect the 'analyze' method
    #     analyze_sig = inspect.signature(Cls.analyze)
    #     params = analyze_sig.parameters
        
    #     # Show container if analyzer accepts 'rest' or 'references'
    #     if 'rest' in params or 'references' in params:
    #         self.ref_signal_container.show()
    #         self._populate_ref_signals()
    #     else:
    #         self.ref_signal_container.hide()

    #     params = Registry.get_analyzer_params(name, path)
    #     for arg, val in params.items():
    #         if arg in ['self', 'kwargs']: continue
    #         lbl = QLabel(arg); le = QLineEdit(str(val))
    #         self.form_layout.addRow(lbl, le); self.inputs[arg] = le

    def on_analyzer_changed(self):
        if sip.isdeleted(self) or self.combo_analyzer.count() == 0: return
        if QApplication.focusObject():
            QApplication.focusObject().clearFocus()

        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                # Hide first to trigger a focus change safely
                item.widget().hide()
                item.widget().deleteLater()
        self.inputs = {}

        try:
            name = self.combo_analyzer.currentText()
            path = self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            
            # 2. Inspect signature for 'rest' or 'references'
            import inspect
            sig = inspect.signature(Cls.analyze)
            has_ref_param = 'rest' in sig.parameters or 'references' in sig.parameters
            
            # 3. Toggle Visibility of the Form Row
            self.lbl_ref_title.setVisible(has_ref_param)
            self.combo_ref_signals.setVisible(has_ref_param)
            
            if has_ref_param:
                self._populate_ref_signals()

            # 4. Build dynamic parameter fields for __init__
            params = Registry.get_analyzer_params(name, path)
            for arg, val in params.items():
                if arg in ['self', 'kwargs']: continue
                le = QLineEdit(str(val))
                self.form_layout.addRow(QLabel(f"<i>{arg}</i>:"), le)
                self.inputs[arg] = le
                
        except Exception as e:
            print(f"[EDA Tab] Error changing analyzer: {e}")


    def run_analysis(self):
        if not self.session.current_data: return

        if self.form_widget.focusWidget():
            self.form_widget.focusWidget().clearFocus()

        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # 1. Get the primary data object based on the FOCUS selector
        focus_key = self.combo_focus.currentText()
        if focus_key not in self.session.workspace:
            print("[EDA] Error: Focus signal not found.")
            return
            
        target_data = self.session.workspace[focus_key]
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:

            # 2. Instantiate Analyzer
            kwargs = {k: ast.literal_eval(v.text()) if v.text().strip() else "" 
                      for k, v in self.inputs.items()}
            Cls = import_class(self.combo_analyzer.currentText(), self.combo_analyzer.currentData())
            analyzer = Cls(**kwargs)

            # 3. Determine Reference Data
            import inspect
            sig = inspect.signature(Cls.analyze)
            selected_names = self.combo_ref_signals.get_checked_items()
            selected_objs = [self.session.workspace[name] for name in selected_names]

            # 4. Map Arguments
            call_kwargs = {}
            if 'rest' in sig.parameters:
                call_kwargs['rest'] = selected_objs[0] if selected_objs else None
            if 'references' in sig.parameters:
                call_kwargs['references'] = selected_objs

            # 5. EXECUTE on the Target Data
            report = analyzer.analyze(target_data, **call_kwargs)
            
            # 6. Save and Show Results
            self.session.eda_report = report
            name = self.combo_analyzer.currentText()
            self.history_data.append({'name': f"{name} #{self.list_history.count()+1}", 'report': report})
            self.list_history.addItem(self.history_data[-1]['name'])
            
            self.show_results(report)
            self.setCurrentIndex(1) # Jump to results tab
            if hasattr(self.window(), 'refresh_gui'): self.window().refresh_gui()

        except Exception as e:
            self.txt_report.setText(f"Analysis Error:\n{e}")
            self.setCurrentIndex(1)
        finally:
             # Standard way
            QApplication.restoreOverrideCursor()
            
            # Hard reset if it's still stuck (clears the whole stack)
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

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
    
    # def _populate_ref_signals(self):
    #     """Fills the list widget with checkable signals from the workspace."""
    #     if sip.isdeleted(self): return
    #     if not self.session or not self.session.workspace:
    #         self.list_ref_signals.clear()
    #         return

    #     # 1. BLOCK SIGNALS: This prevents the 'clear' and 'addItem' from 
    #     # triggering any unintended UI updates while we are building the list.
    #     self.list_ref_signals.blockSignals(True)
        
    #     try:
    #         self.list_ref_signals.clear()
            
    #         # 2. Get keys safely
    #         raw_keys = [k for k, v in self.session.workspace.items() 
    #                     if isinstance(v, mne.io.BaseRaw)]
            
    #         for key in raw_keys:
    #             # 3. Create item safely
    #             item = QListWidgetItem(key)
    #             # Set the checkbox flag
    #             item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    #             item.setCheckState(Qt.CheckState.Unchecked)
    #             self.list_ref_signals.addItem(item)
                
    #     except Exception as e:
    #         print(f"[EDA Tab] Error populating reference signals: {e}")
    #     finally:
    #         # 4. ALWAYS UNBLOCK SIGNALS
    #         self.list_ref_signals.blockSignals(False)

    def _populate_ref_signals(self):

        """Fills the compact combo with checkable signals from the workspace."""
        self.combo_ref_signals.blockSignals(True)
        self.combo_ref_signals.clear()
        
        # Get raw signals from session
        raw_keys = [k for k, v in self.session.workspace.items() 
                    if isinstance(v, mne.io.BaseRaw)]
        
        for key in raw_keys:
            self.combo_ref_signals.addItem(key)
            
        self.combo_ref_signals.setEditText("Select references...")
        self.combo_ref_signals.blockSignals(False)