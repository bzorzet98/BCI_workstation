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

        # 2. Sync Analyzer Logic and Existing Reports
        self.on_analyzer_changed()
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    def on_analyzer_changed(self):
        """Inspects the selected analyzer class and rebuilds the parameter form."""
        if sip.isdeleted(self) or self.combo_analyzer.count() == 0: return
        
        self._clear_parameter_form()

        try:
            name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            
            # Toggle Reference Row based on .analyze() signature
            sig = inspect.signature(Cls.analyze)
            has_ref = any(p in sig.parameters for p in ['rest', 'references'])
            self.lbl_ref_title.setVisible(has_ref)
            self.combo_ref_signals.setVisible(has_ref)

            # Build Kwargs for __init__
            self._build_dynamic_args(name, path)
                
        except Exception as e:
            print(f"[EDA] Introspection Error: {e}")

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
            # 1. Instantiate Analyzer once
            Cls = import_class(self.combo_analyzer.currentText(), self.combo_analyzer.currentData())
            kwargs = {k: ast.literal_eval(v.text()) if v.text().strip() else "" for k, v in self.inputs.items()}
            analyzer = Cls(**kwargs)

            # 2. Execute and Merge
            report = self._execute_multi_context(analyzer, selected_focus)
            
            # 3. Finalize
            self.session.eda_report = report
            self._add_to_history(report)
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
            
            all_frames.append(pd.DataFrame(records))

        # 3. Merge and Package
        master_df = pd.concat(all_frames, ignore_index=True)
        print(f"[EDA] Master DF created. Total rows: {len(master_df)}. Channels: {master_df['Channel'].unique()}")

        return {
            "summary": "\n---\n".join(summaries),
            "df": master_df,
            "visuals": report.get("visuals", ["hist", "scatter", "mapping", "inspect"]),
            "artifacts": {
                "tmin": report['artifacts'].get('tmin', 0),
                "tmax": report['artifacts'].get('tmax', 1)
            },
            "inspection_data": {context_key: report.get('inspection_data', {}) for context_key in focus_keys}
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
            w.cb_metric = QComboBox()
            h_ctrl.addWidget(QLabel("Metric:"))
            h_ctrl.addWidget(w.cb_metric)
            w.cb_metric.currentIndexChanged.connect(callback)
            w.chk_segregate = QCheckBox("Segregate Cues")
            w.chk_segregate.setToolTip("Show individual trial types (Left, Right, etc.)")
            w.chk_segregate.stateChanged.connect(callback)
            h_ctrl.addWidget(w.chk_segregate)
            
        if mode == "scatter":
            w.cb_metric_x = QComboBox()
            w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X:"))
            h_ctrl.addWidget(w.cb_metric_x)
            h_ctrl.addWidget(QLabel("Y:"))
            h_ctrl.addWidget(w.cb_metric_y)
            w.cb_metric_x.currentIndexChanged.connect(callback)
            w.cb_metric_y.currentIndexChanged.connect(callback)

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
        """
        Updates selectors and dynamically shows/hides tabs based on 
        the Master DataFrame contents.
        """
        if not report or 'df' not in report:
            print("[EDA] Error: Report missing Master DataFrame.")
            return

        self.last_report = report
        self.txt_report.setText(report.get('summary', ""))
        
        # 1. DYNAMIC TAB VISIBILITY
        supported_plots = report.get("visuals", ["hist", "scatter", "mapping", "inspect"])
        
        # We assume the tab order: [0:Summary, 1:Hist, 2:Scatter, 3:Mapping, 4:Inspect]
        self.res_subtabs.setTabVisible(1, "hist" in supported_plots)
        self.res_subtabs.setTabVisible(2, "scatter" in supported_plots)
        self.res_subtabs.setTabVisible(3, "mapping" in supported_plots)
        self.res_subtabs.setTabVisible(4, "inspect" in supported_plots)

        # 2. EXTRACT NAMES FROM DATAFRAME
        df = report['df']

        # Channels are simply the unique values in the 'channel' column
        chans = sorted(df['channel'].unique().tolist())
        
        # Metrics are the columns that are NOT metadata or units
        metadata_cols = ['context', 'cue', 'channel', 'trial_idx', 'is_flagged']
        metrics = [col for col in df.columns if col not in metadata_cols and not col.endswith('_unit')]

        # 3. POPULATE SELECTORS
        # We update every tab to ensure they are in sync
        for tab in [self.tab_hist, self.tab_scatter, self.tab_mapping, self.tab_inspect]:
            # Use our robust helper to fill the combo boxes
            self._update_tab_selectors(tab, metrics, chans)

        # 4. INITIAL FOCUS
        # Move the user to the most relevant tab and trigger the first draw
        if "hist" in supported_plots:
            self.res_subtabs.setCurrentIndex(1)
            # Small delay to allow the GUI to process visibility changes before drawing
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.update_hist_canvas)
        else:
            self.res_subtabs.setCurrentIndex(0)
            
        print(f"[EDA] UI Updated: {len(metrics)} metrics and {len(chans)} channels ready.")

    def _update_tab_selectors(self, tab, metrics, chans):
        """
        Populates the combo boxes inside a specific plot sub-tab.
        """
        # Define which attributes to look for in the tab object
        # (These were created in _create_plot_subtab)
        combos_to_fill = [
            (getattr(tab, 'cb_ch', None), chans, True),       # (object, data, add_overlay_option)
            (getattr(tab, 'cb_metric', None), metrics, False),
            (getattr(tab, 'cb_metric_x', None), metrics, False),
            (getattr(tab, 'cb_metric_y', None), metrics, False)
        ]

        for cb, items, add_overlay in combos_to_fill:
            if cb is not None:
                cb.blockSignals(True)
                cb.clear()
                
                if add_overlay:
                    cb.addItem("All Channels (Overlay)")
                
                cb.addItems(items)
                
                # Set a sensible default (first item)
                if cb.count() > 0:
                    cb.setCurrentIndex(0)
                    
                cb.blockSignals(False)

        if hasattr(tab, 'cb_inspect_ctx'):
            tab.cb_inspect_ctx.blockSignals(True)
            tab.cb_inspect_ctx.clear()
            
            # Find which contexts have inspection data in the report
            insp_data = self.last_report.get('inspection_data', {})
            available_contexts = list(insp_data.keys())
            
            tab.cb_inspect_ctx.addItems(available_contexts)
            tab.cb_inspect_ctx.blockSignals(False)

        # Update SpinBox range based on the SELECTED context in inspection
        self._update_inspection_range()
        
    def _update_inspection_range(self):
        """Adjusts the SpinBox maximum based on the selected context's trial count."""
        w = self.tab_inspect
        if not hasattr(w, 'cb_inspect_ctx') or not self.last_report: return
        
        ctx_name = w.cb_inspect_ctx.currentText()
        insp_map = self.last_report.get('inspection_data', {})
        
        if ctx_name in insp_map:
            # Use the length of the first available raw trace as trial count
            first_ch = list(insp_map[ctx_name]['raw_traces'].keys())[0]
            n_trials = len(insp_map[ctx_name]['raw_traces'][first_ch])
            w.sb_trial.setRange(0, max(0, n_trials - 1))
            
    def _add_to_history(self, report):
        name = self.combo_analyzer.currentText()
        self.history_data.append({'name': f"{name} #{len(self.history_data)+1}", 'report': report})
        self.list_history.addItem(self.history_data[-1]['name'])

    def _load_from_history(self, item):
        idx = self.list_history.row(item)
        self.last_report = self.history_data[idx]['report']
        self.show_results(self.last_report)
        self.setCurrentIndex(1)

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0:
            self.history_data.pop(row)
            self.list_history.takeItem(row)

    def update_hist_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_hist; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        
        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        # FILTER THE MASTER DATAFRAME
        df = self.last_report['df']
        if ch != "All Channels (Overlay)":
            df = df[df['channel'] == ch]

        # Apply GUI Math (Log/Z)
        data_to_plot = self._apply_transforms(df[metric].values, w)
        df = df.copy(); df[metric] = data_to_plot # Update copy for seaborn

        hue_col = 'cue' if (hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()) else 'context'

        sns.histplot(data=df, x=metric, hue=hue_col, palette=self.event_colors, 
                     ax=ax, kde=True, element="step", common_norm=False)
        
        ax.set_title(f"{metric} Distribution ({ch})")
        w.canvas.draw()
        
    def update_scatter_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_scatter; w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        if not mx or not my or not ch: return

        df = self.last_report['df']
        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]

        # Apply Math
        df = df.copy()
        df[mx] = self._apply_transforms(df[mx].values, w)
        df[my] = self._apply_transforms(df[my].values, w)

        sns.scatterplot(data=df, x=mx, y=my, hue='context', palette=self.event_colors, ax=ax, alpha=0.5)
        
        # Flagging using the Is_Flagged column we created in _execute
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty:
            ax.scatter(flagged[mx], flagged[my], color='red', marker='x', s=80)

        w.canvas.draw()

    def update_mapping_canvas(self):
        """
        Plots Trial Sequence vs Metric. 
        Uses the trial_idx to show a continuous timeline.
        """
        if not self.last_report or 'df' not in self.last_report: return
        
        w = self.tab_mapping
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        
        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        # 1. Filter Data
        df = self.last_report['df']
        if ch != "All Channels (Overlay)":
            df = df[df['channel'] == ch]

        if df.empty: return

        # 2. Apply Math
        df = df.copy()
        df[metric] = self._apply_transforms(df[metric].values, w)

        # 3. Plot with Seaborn
        sns.scatterplot(
            data=df, x='trial_idx', y=metric, hue='context',
            palette=self.event_colors, ax=ax, s=40, alpha=0.6, edgecolor=None
        )

        # 4. Highlight Rejections (Red X)
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty:
            ax.scatter(flagged['trial_idx'], flagged[metric], 
                       color='red', marker='x', s=80, label="Flagged", zorder=5)

        ax.set_title(f"Sequence Mapping: {metric} ({ch})")
        ax.set_xlabel("Trial Index")
        ax.set_ylabel(f"Value [{self.last_report['df'].get(f'{metric}_unit', ['a.u.'])[0]}]")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        
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