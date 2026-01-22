import ast
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox,QSpinBox,QGroupBox,QListWidgetItem)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
import pandas as pd
import inspect
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mne
from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox

import ast
import numpy as np
import inspect
import mne
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox

class EDATab(QTabWidget):
    """
    Exploratory Data Analysis (EDA) Tab.
    Provides a framework for artifact detection and feature comparison 
    across multiple signal contexts (e.g., Rest vs. MI).
    """
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # --- UI AESTHETICS ---
        self._initialize_aesthetics()

        # --- STATE PERSISTENCE ---
        self.inputs = {}         # Stores QLineEdit references for analyzer params
        self.history_data = []   # List of report dicts
        self.last_report = None  # Pointer to current active report

        self._init_ui()

    def _initialize_aesthetics(self):
        """Sets up theme using Seaborn palettes."""
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")
        
        # Use Seaborn's 'husl' or 'muted' palette for contexts
        # This returns a list of RGB tuples
        self.event_colors = sns.color_palette("husl", 8) 
        
        # Set Seaborn global context for professional font scaling
        sns.set_context("paper", font_scale=1.2)
    # --- UI CONSTRUCTION ---

    def _init_ui(self):
        """Constructs the two primary tabs: Configuration and Results."""
        self.tab_config = QWidget()
        self._build_config_ui()
        
        self.tab_results = QWidget()
        self._build_results_ui()

        self.addTab(self.tab_config, "EDA Configuration")
        self.addTab(self.tab_results, "Diagnostic Results")
        
        # Trigger initial analyzer setup
        self.on_analyzer_changed()

    def _build_config_ui(self):
        """Builds the Configuration tab: [History | Selectors & Params]."""
        main_layout = QHBoxLayout(self.tab_config)

        # 1. Left Sidebar: History
        main_layout.addWidget(self._create_history_panel())

        # 2. Right Panel: Selectors and Dynamic Parameters
        params_panel = QWidget()
        params_vbox = QVBoxLayout(params_panel)
        
        # Top Form (Focus, Analyzer, References)
        self.f_selectors = QFormLayout()
        self._setup_selectors()
        params_vbox.addLayout(self.f_selectors)

        # Dynamic Parameters Scroll Area
        params_vbox.addWidget(QLabel("<b>Analyzer Parameters:</b>"))
        params_vbox.addWidget(self._create_parameter_scroll())
        
        # Action Button
        self.btn_run = QPushButton("Run Analysis / Comparison")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_vbox.addWidget(self.btn_run)
        
        main_layout.addWidget(params_panel)

    def _create_history_panel(self) -> QFrame:
        """Creates the internal history list sidebar."""
        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet(f"border-right: 1px solid {self.c_spine};")
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("<b>Analysis History:</b>"))
        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self._load_from_history)
        layout.addWidget(self.list_history)
        
        btn_del = QPushButton("Remove Entry")
        btn_del.setStyleSheet("background-color: #e74c3c; color: white;")
        btn_del.clicked.connect(self._remove_selected_history)
        layout.addWidget(btn_del)
        return panel

    def _setup_selectors(self):
        """Configures the primary drop-downs for context and analyzer selection."""
        # Multi-Focus Selection (for context comparison)
        self.combo_focus = CheckableComboBox()
        self.combo_focus.setPlaceholderText("Select signal(s) to analyze...")
        
        # Analyzer Selection
        self.combo_analyzer = QComboBox()
        for name, path in Registry.discover_eda_analyzers().items():
            self.combo_analyzer.addItem(name, userData=path)
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)

        # Multi-Reference Selection (for baselines like Rest)
        self.lbl_ref_title = QLabel("<b>Reference Signal(s):</b>")
        self.combo_ref_signals = CheckableComboBox()
        self.combo_ref_signals.setPlaceholderText("Optional baselines...")

        self.f_selectors.addRow("<b>Focus Signal(s):</b>", self.combo_focus)
        self.f_selectors.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)
        self.f_selectors.addRow(self.lbl_ref_title, self.combo_ref_signals)

    def _create_parameter_scroll(self) -> QScrollArea:
        """Creates the scrollable area where dynamic parameters are injected."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background-color: {self.c_bg}; border: 1px solid {self.c_spine};")
        
        self.form_widget = QWidget()
        self.form_layout = QFormLayout(self.form_widget)
        scroll.setWidget(self.form_widget)
        return scroll

    # --- LOGIC & CONTROLLERS ---

    def refresh(self):
        """Synchronizes the UI with the latest Session workspace."""
        if sip.isdeleted(self): return
        
        # 1. Update Focus and Reference lists
        all_signals = list(self.session.workspace.keys())
        raw_signals = [k for k, v in self.session.workspace.items() if isinstance(v, mne.io.BaseRaw)]

        for combo, data in [(self.combo_focus, all_signals), (self.combo_ref_signals, raw_signals)]:
            combo.blockSignals(True)
            combo.clear()
            for item in data: combo.addItem(item)
            combo.blockSignals(False)

        # 2. SMART PARAMETER PRESERVATION
        # If we already have parameters loaded (inputs dict not empty), 
        # do NOT call on_analyzer_changed. Only do it if it's the first time.
        if not self.inputs:
            self.on_analyzer_changed()
        else:
            # We only refresh the reference visibility logic 
            # without clearing the user's typed text
            self._update_reference_visibility_only()
        
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    def _update_reference_visibility_only(self):
        """Helper to toggle Reference row without wiping the parameter form."""
        try:
            name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            sig = inspect.signature(Cls.analyze)
            has_ref = any(p in sig.parameters for p in ['rest', 'references'])
            self.lbl_ref_title.setVisible(has_ref)
            self.combo_ref_signals.setVisible(has_ref)
            if has_ref: self._populate_ref_signals()
        except: pass

    def _populate_ref_signals(self):
        """
        Fills the compact combo with checkable signals from the workspace.
        This allows selecting multiple signals to act as baselines (e.g., Rest).
        """
        if sip.isdeleted(self): return
        
        # 1. Block signals to prevent unintended UI refreshes during population
        self.combo_ref_signals.blockSignals(True)
        self.combo_ref_signals.clear()
        
        # 2. Extract only Raw signals from the workspace
        # (Usually, baselines are raw signals, not epochs)
        raw_keys = [k for k, v in self.session.workspace.items() 
                    if isinstance(v, mne.io.BaseRaw)]
        
        # 3. Add each signal to the checkable combo
        for key in raw_keys:
            self.combo_ref_signals.addItem(key)
            
        # 4. Update the display text
        self.combo_ref_signals.setEditText("Select references (optional)...")
        self.combo_ref_signals.blockSignals(False)
        
    def on_analyzer_changed(self):
        """
        Triggered when the user changes the Target Analyzer.
        Wipes the previous parameter form and builds a new one dynamically.
        """
        # Safety check: ensure widget and data exist
        if sip.isdeleted(self) or self.combo_analyzer.count() == 0: 
            return
        
        # 1. SAFE LAYOUT CLEARING
        # Clear focus first to prevent 'nullptr' focus change crashes in Qt
        if QApplication.focusObject(): 
            QApplication.focusObject().clearFocus()
            
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        
        # Reset the dictionary that stores references to the QLineEdits
        self.inputs = {}

        # 2. ANALYZER INSPECTION (References visibility)
        try:
            name = self.combo_analyzer.currentText()
            path = self.combo_analyzer.currentData()
            
            # Import the class to check its methods
            Cls = import_class(name, path)
            
            # Use introspection to see if .analyze() accepts 'rest' or 'references'
            sig = inspect.signature(Cls.analyze)
            has_ref_param = any(p in sig.parameters for p in ['rest', 'references'])
            
            # Toggle visibility of the Reference row in the selectors form
            self.lbl_ref_title.setVisible(has_ref_param)
            self.combo_ref_signals.setVisible(has_ref_param)
            
            if has_ref_param:
                self._populate_ref_signals()

            # 3. DYNAMIC PARAMETER BUILDING (__init__ arguments)
            # Get default arguments and values from the Registry
            params = Registry.get_analyzer_params(name, path)
            
            for arg, val in params.items():
                # Skip internal Python/Class arguments
                if arg in ['self', 'kwargs', 'args']: 
                    continue
                
                # Create the Label (italicized for aesthetic)
                lbl = QLabel(f"<i>{arg}</i>:")
                lbl.setStyleSheet(f"color: {self.c_text};")
                
                # Create the Input Box (initialized with default value)
                le = QLineEdit(str(val))
                le.setMinimumHeight(25)
                
                # Add to the form layout
                self.form_layout.addRow(lbl, le)
                
                # Store reference in self.inputs so run_analysis can read it
                self.inputs[arg] = le
                
        except Exception as e:
            print(f"[EDA Tab] Error rebuilding analyzer form: {e}")

    def _clear_parameter_form(self):
        """Wipes the dynamic form layout safely."""
        if QApplication.focusObject(): QApplication.focusObject().clearFocus()
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self.inputs = {}

    def _build_dynamic_args(self, name, path):
        """Uses Registry to find default arguments and create UI inputs."""
        params = Registry.get_analyzer_params(name, path)
        for arg, val in params.items():
            if arg in ['self', 'kwargs']: continue
            le = QLineEdit(str(val))
            self.form_layout.addRow(QLabel(f"<i>{arg}</i>:"), le)
            self.inputs[arg] = le

    # --- EXECUTION ENGINE ---

    def run_analysis(self):
        """Primary execution loop. Handles single or multi-context comparison."""
        selected_focus = self.combo_focus.get_checked_items()
        if not selected_focus: return

        # UI State: Spinner and Focus Clear
        if self.form_widget.focusWidget(): self.form_widget.focusWidget().clearFocus()
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            # 1. Capture current parameters from the GUI
            # We save them as strings exactly as they are in the QLineEdits
            current_params = {k: v.text() for k, v in self.inputs.items()}
            analyzer_idx = self.combo_analyzer.currentIndex()
            
            # 2. Instantiate and run (Standard logic)
            kwargs = {k: ast.literal_eval(v) if v.strip() else "" for k, v in current_params.items()}

            # 1. Instantiate Analyzer once
            Cls = import_class(self.combo_analyzer.currentText(), self.combo_analyzer.currentData())
            analyzer = Cls(**kwargs)

            # 2. Execute and Merge
            report = self._execute_multi_context(analyzer, selected_focus)

            # 3. Finalize
            self.session.eda_report = report
            self._add_to_history(report, current_params, analyzer_idx)
            self.show_results(report)
            self.setCurrentIndex(1) # Jump to results

        except Exception as e:
            print(f"[EDA] Analysis Error: {e}")
        finally:
            self._restore_cursor()
    
    def _execute_multi_context(self, analyzer, focus_keys) -> dict:
        """
        Executes analysis and produces a single Master DataFrame.
        """
        all_frames = []
        summaries = []
        
        ref_names = self.combo_ref_signals.get_checked_items()
        ref_objs = [self.session.workspace[n] for n in ref_names]

        inspection_data = {}
        for context_key in focus_keys:
            data = self.session.workspace[context_key]
            
            # 1. Execute Analyzer
            sig = inspect.signature(analyzer.analyze)
            call_kwargs = {}
            if 'rest' in sig.parameters: call_kwargs['rest'] = ref_objs[0] if ref_objs else None
            if 'references' in sig.parameters: call_kwargs['references'] = ref_objs

            report = analyzer.analyze(data, **call_kwargs)
            summaries.append(f"[{context_key}]\n{report['summary']}")

            # 2. Extract Data for DataFrame
            metrics_dict = report['metrics']
            cues = report['artifacts'].get('event_types', [])
            
            # Identify all channels present in the metrics
            # We look at the first metric to find the channel list
            first_metric = list(metrics_dict.keys())[0]
            chans_in_report = list(metrics_dict[first_metric]['values'].keys())
            
            print(f"[EDA] Processing {context_key}: Found channels {chans_in_report}")

            records = []
            for i, cue_id in enumerate(cues):
                # CHECK: Is this trial flagged?
                is_bad_trial = i in report['artifacts']['bad_indices']
                
                for ch in chans_in_report:
                    # Determine if THIS specific channel was flagged
                    is_ch_flagged = False
                    if is_bad_trial:
                        pos = report['artifacts']['bad_indices'].index(i)
                        tags = report['artifacts']['bad_tags'][pos]
                        is_ch_flagged = any(ch.lower() in t.lower() for t in tags)

                    row = {
                        'context': context_key,
                        'cue': f"{context_key}_{cue_id}",
                        'channel': ch,
                        'trial_idx': i,
                        'is_flagged': is_ch_flagged
                    }
                    
                    # Add all metrics for this channel/trial
                    for m_name, m_body in metrics_dict.items():
                        row[m_name] = m_body['values'][ch][i]
                    
                    records.append(row)
            
            inspection_data[context_key] = report.get('inspection_data', {})
            all_frames.append(pd.DataFrame(records))

        # 3. Merge and Package
        master_df = pd.concat(all_frames, ignore_index=True)
        print(f"[EDA] Master DF created. Total rows: {len(master_df)}. Channels: {master_df['channel'].unique()}")

        return {
            "summary": "\n---\n".join(summaries),
            "df": master_df,
            "visuals": report.get("visuals", []),
            "artifacts": {
                "tmin": report['artifacts'].get('tmin', 0),
                "tmax": report['artifacts'].get('tmax', 1)
            },
            "inspection_data": inspection_data
        }
        
    # --- VISUALIZATION & PLOTTING ---
    def _build_results_ui(self):
        """Creates the internal tabs for visualization."""
        layout = QVBoxLayout(self.tab_results)
        self.res_subtabs = QTabWidget()
        
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        self.tab_hist = self._create_plot_subtab(self.update_hist_canvas, mode="hist")
        self.tab_scatter = self._create_plot_subtab(self.update_scatter_canvas, mode="scatter")
        self.tab_mapping = self._create_plot_subtab(self.update_mapping_canvas, mode="mapping")
        self.tab_inspect = self._create_plot_subtab(self.update_inspect_canvas, mode="inspect")

        self.res_subtabs.addTab(self.txt_report, "Text Summary")
        self.res_subtabs.addTab(self.tab_hist, "Histograms")
        self.res_subtabs.addTab(self.tab_scatter, "Feature Scatter")
        self.res_subtabs.addTab(self.tab_mapping, "Trial Mapping")
        self.res_subtabs.addTab(self.tab_inspect, "Inspection")
        layout.addWidget(self.res_subtabs)


    # --- HELPERS ---

    def _create_plot_subtab(self, callback, mode="hist"):
        w = QWidget()
        lay = QVBoxLayout(w)
        h_ctrl = QHBoxLayout()

        # Channel Selector (Present in all modes)
        w.cb_ch = QComboBox() 
        h_ctrl.addWidget(QLabel("CH:"))
        h_ctrl.addWidget(w.cb_ch)
        w.cb_ch.currentIndexChanged.connect(callback)

        # Metric Selectors (Conditional)
        if mode in ["hist", "mapping"]:
            w.cb_inspect_ctx = QComboBox()
            h_ctrl.addWidget(QLabel("Context:"))
            h_ctrl.addWidget(w.cb_inspect_ctx)
            w.cb_inspect_ctx.currentIndexChanged.connect(callback)
            
            w.cb_metric = QComboBox()
            h_ctrl.addWidget(QLabel("Metric:"))
            h_ctrl.addWidget(w.cb_metric)
            w.cb_metric.currentIndexChanged.connect(callback)
            
            w.chk_segregate = QCheckBox("Segregate Cues")
            w.chk_segregate.setToolTip("Show individual trial types (Left, Right, etc.)")
            w.chk_segregate.stateChanged.connect(callback)
            
            h_ctrl.addWidget(w.chk_segregate)
            
        if mode == "scatter":
            w.cb_inspect_ctx = QComboBox()
            h_ctrl.addWidget(QLabel("Context:"))
            h_ctrl.addWidget(w.cb_inspect_ctx)
            w.cb_inspect_ctx.currentIndexChanged.connect(callback)
            
            w.cb_metric_x = QComboBox()
            w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X:"))
            h_ctrl.addWidget(w.cb_metric_x)
            h_ctrl.addWidget(QLabel("Y:"))
            h_ctrl.addWidget(w.cb_metric_y)
            
            w.cb_metric_x.currentIndexChanged.connect(callback)
            w.cb_metric_y.currentIndexChanged.connect(callback)
            
            w.chk_segregate = QCheckBox("Segregate Cues")
            w.chk_segregate.setToolTip("Show individual trial types (Left, Right, etc.)")
            w.chk_segregate.stateChanged.connect(callback)
            
            h_ctrl.addWidget(w.chk_segregate)

        if mode == "inspect":
            w.cb_inspect_ctx = QComboBox()
            h_ctrl.addWidget(QLabel("Context:"))
            h_ctrl.addWidget(w.cb_inspect_ctx)
            w.cb_inspect_ctx.currentIndexChanged.connect(callback)
            
            w.sb_trial = QSpinBox()
            w.sb_trial.setPrefix("Trial #: ")
            w.sb_trial.setRange(0, 99999)
            w.sb_trial.valueChanged.connect(callback)
            
            h_ctrl.addWidget(w.sb_trial)
            
        if mode != "inspect":
            w.chk_log = QCheckBox("Log"); w.chk_z = QCheckBox("Z"); w.chk_mad = QCheckBox("MAD")
            for c in [w.chk_log, w.chk_z, w.chk_mad]: c.stateChanged.connect(callback); h_ctrl.addWidget(c)
            
        h_ctrl.addStretch(); lay.addLayout(h_ctrl)
        w.fig = Figure(facecolor=self.c_bg, tight_layout=True); w.canvas = FigureCanvas(w.fig)
        lay.addWidget(w.canvas); return w

    def _style_ax(self, ax):
        """Applies Seaborn-style aesthetics to Matplotlib axis."""
        ax.set_facecolor(self.c_ax)
        
        # Grid style
        ax.grid(True, linestyle='--', alpha=0.1, color=self.c_text)
        
        # Spine and Tick management
        ax.tick_params(colors=self.c_text, labelsize=9)
        for s in ax.spines.values(): 
            s.set_color(self.c_spine)
            s.set_linewidth(0.5)
            
        ax.xaxis.label.set_color(self.c_text)
        ax.yaxis.label.set_color(self.c_text)
        ax.title.set_color(self.c_text)
        ax.title.set_weight('bold')
        
        # Remove top and right spines for a modern look
        sns.despine(ax=ax)
        
    def _apply_transforms(self, data, sub_tab):
        """Applies Log/Normalization math based on UI state."""
        res = np.array(data, dtype=float)
        if hasattr(sub_tab, 'chk_log') and sub_tab.chk_log.isChecked(): res = np.log10(np.maximum(res, 1e-12))
        if hasattr(sub_tab, 'chk_z') and sub_tab.chk_z.isChecked(): res = (res - np.mean(res)) / (np.std(res) + 1e-9)
        elif hasattr(sub_tab, 'chk_mad') and sub_tab.chk_mad.isChecked():
            med = np.median(res); mad = np.median(np.abs(res - med))
            res = (res - med) / (mad * 1.4826 + 1e-9)
        return res

    def _restore_cursor(self):
        """Clears the cursor spinner stack."""
        while QApplication.overrideCursor() is not None: QApplication.restoreOverrideCursor()

    def show_results(self, report):
        if not report or 'df' not in report: return
        self.last_report = report
        self.txt_report.setText(report.get('summary', ""))
        
        # 1. VISIBILITY LOGIC
        supported = report.get("visuals", ["hist", "scatter", "mapping", "inspect"])
        tabs_to_update = []
        
        mapping = {"hist": (1, self.tab_hist), "scatter": (2, self.tab_scatter), 
                   "mapping": (3, self.tab_mapping), "inspect": (4, self.tab_inspect)}

        for tag, (idx, tab_obj) in mapping.items():
            is_visible = tag in supported
            self.res_subtabs.setTabVisible(idx, is_visible)
            if is_visible: tabs_to_update.append(tab_obj)

        # 2. NAMES EXTRACTION
        df = report['df']
        chans = sorted(df['channel'].unique().tolist())
        meta = ['context', 'cue', 'channel', 'trial_idx', 'is_flagged']
        metrics = [c for c in df.columns if c not in meta and not c.endswith('_unit')]

        # 3. TARGETED UPDATE
        for tab in tabs_to_update:
            self._update_tab_selectors(tab, metrics, chans)

        # 4. INITIAL FOCUS
        if "hist" in supported:
            self.res_subtabs.setCurrentIndex(1)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.update_hist_canvas)

    def _update_tab_selectors(self, tab, metrics, chans):
        """
        Populates the combo boxes inside a specific plot sub-tab.
        """
        # 1. Populate Metrics and Channels (Existing logic)
        combos_to_fill = [
            (getattr(tab, 'cb_ch', None), chans, True),
            (getattr(tab, 'cb_metric', None), metrics, False),
            (getattr(tab, 'cb_metric_x', None), metrics, False),
            (getattr(tab, 'cb_metric_y', None), metrics, False)
        ]

        for cb, items, add_overlay in combos_to_fill:
            if cb is not None:
                cb.blockSignals(True)
                cb.clear()
                if add_overlay: cb.addItem("All Channels (Overlay)")
                cb.addItems(items)
                if cb.count() > 0: cb.setCurrentIndex(0)
                cb.blockSignals(False)

        # 2. POPULATE CONTEXT SELECTOR (The new logic)
        if hasattr(tab, 'cb_inspect_ctx'):
            cb_ctx = tab.cb_inspect_ctx
            cb_ctx.blockSignals(True)
            cb_ctx.clear()
            
            # Get available contexts from the Master DataFrame
            df = self.last_report['df']
            available_contexts = sorted(df['context'].unique().tolist())
            
            # --- LOGIC: 'All Contexts' vs Single Context ---
            # If the current tab is NOT the inspection tab, add the "All" option
            if tab != self.tab_inspect:
                cb_ctx.addItem("All Contexts (Comparison)")
            
            cb_ctx.addItems(available_contexts)
            
            if cb_ctx.count() > 0: cb_ctx.setCurrentIndex(0)
            cb_ctx.blockSignals(False)

        # 3. Update SpinBox range (only if on inspection tab)
        if tab == self.tab_inspect:
            self._update_inspection_range()
        
    def _update_inspection_range(self):
        """Adjusts the SpinBox range specifically for the Inspection tab."""
        if not self.last_report or not hasattr(self.tab_inspect, 'cb_inspect_ctx'): 
            return
        
        tab = self.tab_inspect
        ctx_name = tab.cb_inspect_ctx.currentText()
        
        # Inspection doesn't have an "All" option, but safety first
        insp_map = self.last_report.get('inspection_data', {})
        if ctx_name in insp_map:
            # Get the trial count from the traces
            first_ch = list(insp_map[ctx_name].get('raw_traces', {}).keys())[0]
            n_trials = len(insp_map[ctx_name]['raw_traces'][first_ch])
            tab.sb_trial.setRange(0, max(0, n_trials - 1))
            
    def _add_to_history(self, report, current_params, analyzer_idx):
        name = self.combo_analyzer.currentText()
        history_entry = {
            'name': f"{name} #{len(self.history_data)+1}",
            'report': report,
            'params': current_params,       
            'analyzer_idx': analyzer_idx    
        }
        self.history_data.append(history_entry)
        self.list_history.addItem(history_entry['name'])

    def _load_from_history(self, item):
        idx = self.list_history.row(item)
        hist = self.history_data[idx]

        # 1. Restore Analyzer Selection
        self.combo_analyzer.blockSignals(True)
        self.combo_analyzer.setCurrentIndex(hist['analyzer_idx'])
        self.combo_analyzer.blockSignals(False)

        # 2. Rebuild the form (This clears old widgets)
        self.on_analyzer_changed()

        # 3. Inject saved parameters into the NEWLY created QLineEdits
        for key, value in hist['params'].items():
            if key in self.inputs:
                self.inputs[key].setText(str(value))

        # 4. Load report data to plots
        self.last_report = hist['report']
        self.show_results(self.last_report)
        self.setCurrentIndex(1)

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0:
            self.history_data.pop(row)
            self.list_history.takeItem(row)

    def update_hist_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_hist
        
        # 1. GET FILTERED DATA
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data for selection", ha='center', color='gray')
            w.canvas.draw(); return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        # 2. CHANNEL FILTER
        if ch != "All Channels (Overlay)":
            df = df[df['channel'] == ch]

        # 3. MATH TRANSFORMS
        df = df.copy()
        df[metric] = self._apply_transforms(df[metric].values, w)

        # 4. FIXED HUE LOGIC
        # If the user selected a specific context, we don't need 'context' as a hue
        # unless they want to segregate by cues.
        is_segregated = hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()
        is_comparison = "All Contexts" in w.cb_inspect_ctx.currentText()

        if is_segregated:
            hue_col = 'cue'
        elif is_comparison:
            hue_col = 'context'
        else:
            hue_col = None # IMPORTANT: Disable hue for single context selection

        # 5. PLOT
        sns.histplot(
            data=df, x=metric, hue=hue_col, 
            palette=self.event_colors, ax=ax,
            kde=True, element="step", alpha=0.3, common_norm=False,
            stat='probability'
        )
        
        ax.set_title(f"Distribution: {metric} ({ch})")
        w.canvas.draw()
        
    def update_scatter_canvas(self):
        
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_scatter
        
        # 1. GET FILTERED DATA
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data for selection", ha='center', color='gray')
            w.canvas.draw(); return

        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        if not mx or not my or not ch: return

        # 2. CHANNEL FILTER
        if ch != "All Channels (Overlay)":
            df = df[df['channel'] == ch]

        # 3. MATH TRANSFORMS
        df = df.copy()
        df = df.copy()
        df[mx] = self._apply_transforms(df[mx].values, w)
        df[my] = self._apply_transforms(df[my].values, w)

        # 4. FIXED HUE LOGIC
        # If the user selected a specific context, we don't need 'context' as a hue
        # unless they want to segregate by cues.
        is_segregated = hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()
        is_comparison = "All Contexts" in w.cb_inspect_ctx.currentText()

        if is_segregated:
            hue_col = 'cue'
        elif is_comparison:
            hue_col = 'context'
        else:
            hue_col = None # IMPORTANT: Disable hue for single context selection
            
        sns.scatterplot(data=df, x=mx, y=my, hue=hue_col, palette=self.event_colors, ax=ax, alpha=0.5)
        
        # Flagging using the Is_Flagged column we created in _execute
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty:
            ax.scatter(flagged[mx], flagged[my], color='red', marker='x', s=80)

        w.canvas.draw()

    def update_mapping_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_mapping
        
        # 1. GET FILTERED DATA
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data for selection", ha='center', color='gray')
            w.canvas.draw(); return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        # 2. CHANNEL FILTER
        if ch != "All Channels (Overlay)":
            df = df[df['channel'] == ch]

        # 3. MATH TRANSFORMS
        df = df.copy()
        df[metric] = self._apply_transforms(df[metric].values, w)

        # 4. FIXED HUE LOGIC
        # If the user selected a specific context, we don't need 'context' as a hue
        # unless they want to segregate by cues.
        is_segregated = hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()
        is_comparison = "All Contexts" in w.cb_inspect_ctx.currentText()

        if is_segregated:
            hue_col = 'cue'
        elif is_comparison:
            hue_col = 'context'
        else:
            hue_col = None # IMPORTANT: Disable hue for single context selection

        sns.scatterplot(data=df, x='trial_idx', y=metric, hue=hue_col, 
                        palette=self.event_colors, ax=ax, s=40, alpha=0.6)
        
        # 6. --- ADDED: SMART ARTIFACT OVERLAY ---
        # Filter for flagged trials within the current data slice
        flagged_df = df[df['is_flagged'] == True]
        
        if not flagged_df.empty:
            ax.scatter(
                flagged_df['trial_idx'], 
                flagged_df[metric], 
                color='red', 
                marker='x', 
                s=100, 
                linewidths=2,   
                label='Artifact (Flagged)',
                zorder=5 # Ensure crosses are on top of dots
            )

        # 7. AESTHETICS
        unit = self.last_report['df'].get(f"{metric}_unit", ["a.u."])[0]
        ax.set_title(f"Sequence Mapping: {metric} ({ch})")
        ax.set_xlabel("Trial Index")
        ax.set_ylabel(f"Value [{unit}]")
        
        # Adjust legend to include the 'Flagged' marker
        ax.legend(frameon=False, loc='upper right', fontsize=8)
        
        w.canvas.draw()
        
        
    def update_inspect_canvas(self):
        """Microscope view: Filters by Context and Trial ID."""
        if not self.last_report or 'inspection_data' not in self.last_report: return
        
        w = self.tab_inspect
        ctx_name = w.cb_inspect_ctx.currentText()
        ch = w.cb_ch.currentText()
        tid = w.sb_trial.value()
        
        insp_map = self.last_report['inspection_data']
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)

        # 1. Safety Checks
        if ctx_name not in insp_map: return
        
        ctx_data = insp_map[ctx_name]
        if ch == "All Channels (Overlay)" or ch not in ctx_data['raw_traces']:
            ax.text(0.5, 0.5, "Select a specific channel", ha='center', color='gray')
            w.canvas.draw(); return

        try:
            # 2. Extract specific trace
            raw = np.array(ctx_data['raw_traces'][ch][tid])
            env = np.array(ctx_data['envelope_traces'][ch][tid])
            
            t = np.linspace(self.last_report['artifacts'].get('tmin', 0), 
                            self.last_report['artifacts'].get('tmax', 1), len(raw))

            # 3. Plot
            ax.plot(t, raw*1e6, color='gray', alpha=0.4, label="Raw (uV)")
            ax.plot(t, env*1e6, color='#589df6', lw=2, label="Envelope")
            
            thresh = ctx_data.get('threshold_lines', {}).get(ch, 0)
            if thresh != 0:
                ax.axhline(y=thresh*1e6, color='red', ls='--', label="Threshold")
            
            # Check Master DF for the "Flagged" status of this specific trial/context
            df = self.last_report['df']
            is_bad = df[(df['context'] == ctx_name) & 
                        (df['trial_idx'] == tid) & 
                        (df['channel'] == ch)]['is_flagged'].any()
            
            status = "[FLAGGED]" if is_bad else "[CLEAN]"
            ax.set_title(f"{status} {ctx_name} | {ch} | Trial #{tid}")
            ax.legend(frameon=False)
            w.canvas.draw()
            
        except Exception as e:
            print(f"[EDA] Inspection Error: {e}")
            
    def _get_tab_filtered_df(self, tab):
        """Returns the Master DF filtered by the specific tab's context selector."""
        if not self.last_report or 'df' not in self.last_report: return None
        
        df = self.last_report['df']
        
        if not hasattr(tab, 'cb_inspect_ctx'):
            return df
        
        # Get local selection
        selection = tab.cb_inspect_ctx.currentText()
        
        if selection == "All Contexts (Comparison)":
            return df # Return everything
        else:
            return df[df['context'] == selection] # Return only one context
    