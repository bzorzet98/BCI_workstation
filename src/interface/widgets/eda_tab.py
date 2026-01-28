import ast
import inspect
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox, QSpinBox,QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox

class EDATab(QTabWidget):
    """
    Advanced Exploratory Data Analysis (EDA) Tab.
    
    Features:
    - Signature-Driven UI: Automatically creates signal selectors based on analyzer code.
    - Stateful UI: Preserves parameters across subjects and restores them from history.
    - Comparative Analytics: Uses a Master Pandas DataFrame for Seaborn-based comparisons.
    - Unified Inspection: Context-aware raw signal microscope.
    """
    
    TRACE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
                    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    
    LINE_COLORS = ["#A50F15", "#252525", "#54278F", "#006D2C", "#F0027F", 
                   "#BF812D", "#08306B", "#01665E", "#B35806", "#000000"]

    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # --- 1. STATE INITIALIZATION ---
        self.analyzer_bundles = {}   # Stores pre-built UI bundles
        self.history_data = []       
        self.last_report = None      
        
        self._initialize_aesthetics()
        self._init_ui()
        
        # --- 2. PRE-BUILD ALL UIs ---
        self._prebuild_analyzer_uis()

    def _setup_selectors(self):
        """Fixed top-level selectors."""
        # Target Analyzer Selection
        self.combo_analyzer = QComboBox()
        for name, path in Registry.discover_eda_analyzers().items():
            self.combo_analyzer.addItem(name, userData=path)
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)

        # Common Focus Signal (Every analyzer needs this)
        self.lbl_main_signal = QLabel("<b>Target Signal(s):</b>")
        self.combo_focus = CheckableComboBox()

        self.f_selectors.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)
        self.f_selectors.addRow(self.lbl_main_signal, self.combo_focus)
        
        # --- THE STACK: This holds the dynamic parts (Extra Signals + Params) ---
        self.config_stack = QStackedWidget()
        # We add the stack as a spanning row at the bottom of selectors
        self.f_selectors.addRow(self.config_stack)

    def _initialize_aesthetics(self):
        """Sets up the visual theme for plots."""
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")
        self.event_colors = sns.color_palette("husl", 8) 
        sns.set_context("paper", font_scale=1.1)

    # --- UI CONSTRUCTION ---

    def _init_ui(self):
        """Builds the main configuration and results tab structures."""
        self.tab_config = QWidget()
        self._build_config_ui()
        
        self.tab_results = QWidget()
        self._build_results_ui()

        self.addTab(self.tab_config, "EDA Configuration")
        self.addTab(self.tab_results, "Diagnostic Results")
        
        # Initialize the form for the first time
        self.on_analyzer_changed()

    def _build_config_ui(self):
        """Layout: [ History Sidebar | Parameter Setup ]"""
        main_layout = QHBoxLayout(self.tab_config)

        # 1. Left Sidebar: History (Keep existing code)
        main_layout.addWidget(self._create_history_panel())

        # 2. Right Panel: Selectors and Dynamic Parameters
        params_panel = QWidget()
        params_vbox = QVBoxLayout(params_panel)
        
        self.f_selectors = QFormLayout()
        self._setup_selectors()
        params_vbox.addLayout(self.f_selectors)

        params_vbox.addWidget(QLabel("<b>Analyzer Parameters:</b>"))
        
        # --- FIX: Save the scroll area as a class attribute ---
        self.scroll_config = self._create_parameter_scroll()
        params_vbox.addWidget(self.scroll_config)
        
        # Action Button
        self.btn_run = QPushButton("Run Analysis / Comparison")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 4px;")
        self.btn_run.clicked.connect(self.run_analysis)
        params_vbox.addWidget(self.btn_run)
        
        main_layout.addWidget(params_panel)
    def _prebuild_analyzer_uis(self):
        """Creates a dedicated UI bundle for every analyzer in the Registry."""
        available = Registry.discover_eda_analyzers()
        
        for name, path in available.items():
            # 1. Create a container for this analyzer
            bundle_widget = QWidget()
            layout = QVBoxLayout(bundle_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # 2. Section A: Extra Signal Selectors (from analyze signature)
            extra_signals_form = QFormLayout()
            signal_selectors = {}
            
            Cls = import_class(name, path)
            sig = inspect.signature(Cls.analyze)
            params = [p for p in sig.parameters.values() if p.name != 'self']
            
            # Skip first param (it's the common Focus combo), handle the rest
            if len(params) > 1:
                for extra in params[1:]:
                    lbl = QLabel(f"<b>{extra.name.replace('_',' ').title()}:</b>")
                    combo = CheckableComboBox()
                    extra_signals_form.addRow(lbl, combo)
                    signal_selectors[extra.name] = combo
            
            layout.addLayout(extra_signals_form)

            # 3. Section B: Parameter Fields (from __init__)
            layout.addWidget(QLabel("<b>Analyzer Parameters:</b>"))
            params_form = QFormLayout()
            inputs = {}
            defaults = Registry.get_analyzer_params(name, path)
            for arg, val in defaults.items():
                if arg in ['self', 'kwargs', 'args']: continue
                le = QLineEdit(str(val))
                params_form.addRow(QLabel(f"<i>{arg}</i>:"), le)
                inputs[arg] = le
            
            layout.addLayout(params_form)
            layout.addStretch() # Push everything up

            # 4. Store Bundle
            self.analyzer_bundles[name] = {
                "widget": bundle_widget,
                "inputs": inputs,
                "selectors": signal_selectors,
                "main_param_name": params[0].name if params else "data"
            }
            
            # Add to the visual stack
            self.config_stack.addWidget(bundle_widget)
             
    def _create_history_panel(self) -> QFrame:
        """Helper to build the history list sidebar."""
        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet(f"border-right: 1px solid {self.c_spine};")
        layout = QVBoxLayout(panel)
        
        layout.addWidget(QLabel("<b>Analysis History:</b>"))
        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self._load_from_history)
        layout.addWidget(self.list_history)
        
        btn_del = QPushButton("Remove Selected")
        btn_del.setStyleSheet("background-color: #e74c3c; color: white;")
        btn_del.clicked.connect(self._remove_selected_history)
        layout.addWidget(btn_del)
        return panel

    def _create_parameter_scroll(self) -> QScrollArea:
        self.scroll_config = QScrollArea()
        self.scroll_config.setWidgetResizable(True)
        self.scroll_config.setStyleSheet(f"background-color: {self.c_bg}; border: 1px solid {self.c_spine};")
        
        # --- ANCHOR 2: A simple container ---
        self.form_container = QWidget()
        self.scroll_config.setWidget(self.form_container)
        return self.scroll_config

    def _build_results_ui(self):
        """Constructs sub-tabs for summary and Seaborn visualizations."""
        layout = QVBoxLayout(self.tab_results)
        self.res_subtabs = QTabWidget()
        
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        self.tab_hist = self._create_plot_subtab(self.update_hist_canvas, mode="hist")
        self.tab_scatter = self._create_plot_subtab(self.update_scatter_canvas, mode="scatter")
        self.tab_mapping = self._create_plot_subtab(self.update_mapping_canvas, mode="mapping")
        self.tab_inspect = self._create_plot_subtab(self.update_inspect_canvas, mode="inspect")

        self.res_subtabs.addTab(self.txt_report, "Text Summary")
        self.res_subtabs.addTab(self.tab_hist, "Histograms")
        self.res_subtabs.addTab(self.tab_scatter, "Scatters")
        self.res_subtabs.addTab(self.tab_mapping, "Mappings")
        self.res_subtabs.addTab(self.tab_inspect, "Inspection")
        layout.addWidget(self.res_subtabs)

    # --- LOGIC & INTROSPECTION ---

    def refresh(self):
        """Safely synchronizes workspace without wiping typed parameters."""
        if sip.isdeleted(self) or not self.session.workspace: return
        
        # 1. Update dropdown lists without triggering recursive form rebuilds
        self.combo_focus.blockSignals(True)
        for combo in self.signal_selectors.values(): combo.blockSignals(True)
        
        self._populate_all_signal_selectors()
        
        self.combo_focus.blockSignals(False)
        for combo in self.signal_selectors.values(): combo.blockSignals(False)

        # 2. State Preservation: Only rebuild the parameter form if it's currently empty
        if not self.inputs:
            self.on_analyzer_changed()
        
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    def on_analyzer_changed(self):
        if sip.isdeleted(self) or self.combo_analyzer.count() == 0: return
        
        # Block signals to prevent a paint event from firing mid-rebuild
        self.combo_analyzer.blockSignals(True)
        self._clear_dynamic_ui()

        try:
            name, path = self.combo_analyzer.currentText(), self.combo_analyzer.currentData()
            Cls = import_class(name, path)
            analyze_sig = inspect.signature(Cls.analyze)
            params = [p for p in analyze_sig.parameters.values() if p.name != 'self']
            
            if params:
                self.main_param_name = params[0].name
                self.lbl_main_signal.setText(f"<b>{self.main_param_name.replace('_',' ').title()}(s):</b>")
                
                for extra in params[1:]:
                    lbl = QLabel(f"<b>{extra.name.replace('_',' ').title()}:</b>")
                    combo = CheckableComboBox()
                    # ADD TO THE NEW SUB-FORM
                    self.extra_sigs_form.addRow(lbl, combo)
                    self.signal_selectors[extra.name] = combo

            self._populate_all_signal_selectors()
            self._build_dynamic_args(name, path) # This adds to self.form_layout
                
        except Exception as e:
            print(f"[EDA] Rebuild Error: {e}")
        finally:
            self.combo_analyzer.blockSignals(False)

    def _clear_dynamic_ui(self):
        """
        Swaps out the entire container widgets. 
        This prevents the layout engine from ever seeing a 'null' widget during a refresh.
        """
        # 1. Safety Focus
        self.setFocus()

        # 2. NUKE PARAMETERS FORM
        # We take the current widget out of the scroll area
        old_form = self.scroll_config.takeWidget()
        if old_form:
            old_form.deleteLater() # Safe because it's no longer 'visible' to the layout
        
        # Re-create a fresh, clean container
        self.form_container = QWidget()
        self.form_layout = QFormLayout(self.form_container)
        self.scroll_config.setWidget(self.form_container)
        self.inputs = {}

        # 3. NUKE EXTRA SIGNAL SELECTORS
        # We find the anchor in the f_selectors (it's the 3rd row, index 2)
        # We replace the widget at that row
        self.signal_selectors = {}
        
        # We delete the old inner container by replacing it
        # First, find the old one
        old_extra = self.extra_sigs_anchor
        # Create the new one
        self.extra_sigs_anchor = QWidget()
        self.extra_sigs_form = QFormLayout(self.extra_sigs_anchor)
        self.extra_sigs_form.setContentsMargins(0, 0, 0, 0)
        
        # Swap the row in the main form layout
        # (index 2 is the 3rd row: Analyzer, Focus, then Extras)
        self.f_selectors.setWidget(2, QFormLayout.ItemRole.SpanningRole, self.extra_sigs_anchor)
        
        if old_extra:
            old_extra.deleteLater()
        
    def _populate_all_signal_selectors(self):
        """Syncs all dropdowns with the latest workspace keys."""
        all_keys = list(self.session.workspace.keys())
        
        self.combo_focus.blockSignals(True); self.combo_focus.clear()
        for k in all_keys: self.combo_focus.addItem(k)
        self.combo_focus.blockSignals(False)

        for combo in self.signal_selectors.values():
            combo.blockSignals(True); combo.clear()
            for k in all_keys: combo.addItem(k)
            combo.blockSignals(False)

    def _build_dynamic_args(self, name, path):
        """Creates labels and line edits for analyzer settings."""
        defaults = Registry.get_analyzer_params(name, path)
        for arg, val in defaults.items():
            if arg in ['self', 'kwargs', 'args']: continue
            le = QLineEdit(str(val))
            self.form_layout.addRow(QLabel(f"<i>{arg}</i>:"), le)
            self.inputs[arg] = le

    # --- EXECUTION ENGINE ---

    def run_analysis(self):
        """Main execution loop: gathers params, runs analyze(), and merges results."""
        focus_keys = self.combo_focus.get_checked_items()
        if not focus_keys: return

        if self.form_widget.focusWidget(): self.form_widget.focusWidget().clearFocus()
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            # 1. Gather configuration
            kwargs_init = {k: v.text() for k, v in self.inputs.items()} # Save as strings for history
            kwargs_eval = {k: ast.literal_eval(v) if v.strip() else "" for k, v in kwargs_init.items()}
            
            # 2. Gather extra signals
            extra_signals = {}
            for arg_name, combo in self.signal_selectors.items():
                keys = combo.get_checked_items()
                objs = [self.session.workspace[k] for k in keys]
                extra_signals[arg_name] = objs[0] if len(objs) == 1 else (objs if objs else None)

            # 3. EXECUTE & MERGE
            Cls = import_class(self.combo_analyzer.currentText(), self.combo_analyzer.currentData())
            analyzer = Cls(**kwargs_eval)
            report = self._execute_multi_context(analyzer, focus_keys, extra_signals)
            
            # 4. FINALIZE
            self.session.eda_report = report
            self._add_to_history(report, kwargs_init, self.combo_analyzer.currentIndex())
            self.show_results(report)
            self.setCurrentIndex(1)

        except Exception as e:
            print(f"[EDA] Analysis Execution Error: {e}")
        finally:
            while QApplication.overrideCursor(): QApplication.restoreOverrideCursor()

    def _execute_multi_context(self, analyzer, focus_keys, extra_sigs) -> dict:
        """Standardized merging into a Tidy Master DataFrame."""
        all_frames = []
        summaries = []
        inspection_map = {}
        global_counter = 0

        for context_key in focus_keys:
            # Dynamic call using introspection mapping
            call_kwargs = {self.main_param_name: self.session.workspace[context_key]}
            call_kwargs.update(extra_sigs)
            
            report = analyzer.analyze(**call_kwargs)
            summaries.append(f"[{context_key}]\n{report['summary']}")

            # Tidy unrolling
            metrics = report['metrics']
            cues = report['artifacts'].get('event_types', [])
            chans = list(metrics[list(metrics.keys())[0]]['values'].keys())
            
            records = []
            for i, cue_id in enumerate(cues):
                is_bad_trial = i in report['artifacts']['bad_indices']
                for ch in chans:
                    is_ch_flagged = False
                    if is_bad_trial:
                        pos = report['artifacts']['bad_indices'].index(i)
                        is_ch_flagged = any(ch.lower() in tag.lower() for tag in report['artifacts']['bad_tags'][pos])
                    
                    row = {
                        'context': context_key, 'cue': f"{context_key}_{cue_id}", 'channel': ch,
                        'trial_idx': i, 'global_idx': global_counter + i, 'is_flagged': is_ch_flagged,
                        **{m: m_body['values'][ch][i] for m, m_body in metrics.items()}
                    }
                    records.append(row)
            
            all_frames.append(pd.DataFrame(records))
            inspection_map[context_key] = report.get('inspection_data', {})
            global_counter += len(cues)

        return {
            "summary": "\n---\n".join(summaries),
            "df": pd.concat(all_frames, ignore_index=True),
            "visuals": report.get("visuals", ["hist", "scatter", "mapping"]),
            "artifacts": report['artifacts'],
            "inspection_data": inspection_map
        }

    # --- VISUALIZATION ---

    def show_results(self, report):
        """Optimized results display: only updates visible tabs."""
        if not report or 'df' not in report: return
        self.last_report = report
        self.txt_report.setText(report.get('summary', ""))
        
        supported = report.get("visuals", ["hist", "scatter", "mapping", "inspect"])
        tabs_to_update = []
        
        # Visibility Logic
        mapping = {"hist": (1, self.tab_hist), "scatter": (2, self.tab_scatter), 
                   "mapping": (3, self.tab_mapping), "inspect": (4, self.tab_inspect)}

        for tag, (idx, tab_obj) in mapping.items():
            is_visible = tag in supported
            self.res_subtabs.setTabVisible(idx, is_visible)
            if is_visible: tabs_to_update.append(tab_obj)

        # Names extraction
        df = report['df']
        chans = sorted(df['channel'].unique().tolist())
        meta = ['context', 'cue', 'channel', 'trial_idx', 'is_flagged', 'global_idx']
        metrics = [c for c in df.columns if c not in meta and not c.endswith('_unit')]

        # Targeted Update
        for tab in tabs_to_update:
            self._update_tab_selectors(tab, metrics, chans)

        # Focus management
        if "hist" in supported:
            self.res_subtabs.setCurrentIndex(1)
            # Small delay to ensure Seaborn has a valid drawing surface
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.update_hist_canvas)

    def _update_tab_selectors(self, tab, metrics, chans):
        """Populates combo boxes inside specific plot sub-tabs."""
        combos = [(getattr(tab, 'cb_ch', None), chans, True),
                  (getattr(tab, 'cb_metric', None), metrics, False),
                  (getattr(tab, 'cb_metric_x', None), metrics, False),
                  (getattr(tab, 'cb_metric_y', None), metrics, False)]

        for cb, items, overlay in combos:
            if cb:
                cb.blockSignals(True); cb.clear()
                if overlay: cb.addItem("All Channels (Overlay)")
                cb.addItems(items); cb.setCurrentIndex(0); cb.blockSignals(False)

        if hasattr(tab, 'cb_inspect_ctx'):
            cb_ctx = tab.cb_inspect_ctx
            cb_ctx.blockSignals(True); cb_ctx.clear()
            if tab != self.tab_inspect: cb_ctx.addItem("All Contexts (Comparison)")
            cb_ctx.addItems(sorted(self.last_report['df']['context'].unique().tolist()))
            cb_ctx.setCurrentIndex(0); cb_ctx.blockSignals(False)

        if tab == self.tab_inspect: self._update_inspection_range()

    # --- PLOTTING CANVASES ---

    def update_hist_canvas(self):
        """Seaborn HistPlot with KDE and Context Comparison."""
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_hist
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]
        
        df = df.copy(); df[metric] = self._apply_transforms(df[metric].values, w)

        # Smart Hue Logic
        is_segregated = hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()
        is_comp = "All Contexts" in w.cb_inspect_ctx.currentText()
        hue_col = 'cue' if is_segregated else ('context' if is_comp else None)

        try:
            sns.histplot(data=df, x=metric, hue=hue_col, palette=self.event_colors, 
                         ax=ax, kde=True, element="step", common_norm=False, stat='probability')
            ax.set_title(f"Distribution: {metric} ({ch})")
        except: pass
        w.canvas.draw()

    def update_scatter_canvas(self):
        """Seaborn ScatterPlot for metric correlation."""
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_scatter
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        if not mx or not my or not ch: return

        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]
        
        df = df.copy()
        df[mx] = self._apply_transforms(df[mx].values, w)
        df[my] = self._apply_transforms(df[my].values, w)

        hue_col = 'context' if "All Contexts" in w.cb_inspect_ctx.currentText() else None
        sns.scatterplot(data=df, x=mx, y=my, hue=hue_col, palette=self.event_colors, ax=ax, alpha=0.5)
        
        # Flags overlay
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty: ax.scatter(flagged[mx], flagged[my], color='red', marker='x', s=80)
        
        w.canvas.draw()

    def update_mapping_canvas(self):
        """Seaborn ScatterPlot for trial sequence mapping."""
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_mapping
        df = self._get_tab_filtered_df(w)
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return

        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]
        
        df = df.copy(); df[metric] = self._apply_transforms(df[metric].values, w)

        hue_col = 'context' if "All Contexts" in w.cb_inspect_ctx.currentText() else None
        sns.scatterplot(data=df, x='trial_idx', y=metric, hue=hue_col, palette=self.event_colors, ax=ax, s=40)
        
        # Red X for Artifacts
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty: ax.scatter(flagged['trial_idx'], flagged[metric], color='red', marker='x', s=100)
        
        ax.set_title(f"Sequence Mapping: {metric}")
        w.canvas.draw()

    def update_inspect_canvas(self):
        """Microscope view for raw signal traces and detection metadata."""
        if not self.last_report or 'inspection_data' not in self.last_report: return
        w = self.tab_inspect
        self._update_inspection_range()
        
        ctx_name = w.cb_inspect_ctx.currentText()
        ch, tid = w.cb_ch.currentText(), w.sb_trial.value()
        
        insp_map = self.last_report['inspection_data']
        if ctx_name not in insp_map: return
        ctx_data = insp_map[ctx_name]
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if ch == "All Channels (Overlay)" or ch not in ctx_data.get('raw_traces', {}):
            ax.text(0.5, 0.5, "Select specific channel", ha='center', color='gray'); w.canvas.draw(); return

        try:
            t = np.linspace(self.last_report['artifacts'].get('tmin', 0), 
                            self.last_report['artifacts'].get('tmax', 1), 
                            len(ctx_data['raw_traces'][ch][tid]))
            
            idx_t, idx_l = 0, 0
            for key, data in ctx_data.items():
                if key.endswith("_traces"):
                    ax.plot(t, np.array(data[ch][tid]), color=self.TRACE_COLORS[idx_t % 10], label=key.replace("_traces", ""))
                    idx_t += 1
                elif key.endswith("_lines"):
                    for i, val in enumerate(data[ch]):
                        ax.axhline(y=val, color=self.LINE_COLORS[idx_l % 10], ls='--', label=f"{key}_{i}")
                        idx_l += 1
            
            # Check flagged status in DF
            bad = self.last_report['df'][(self.last_report['df']['context'] == ctx_name) & 
                                        (self.last_report['df']['trial_idx'] == tid) & 
                                        (self.last_report['df']['channel'] == ch)]['is_flagged'].any()
            
            ax.set_title(f"{'[FLAGGED]' if bad else '[CLEAN]'} {ctx_name} | {ch} | Trial #{tid}")
            ax.legend(frameon=False, loc='upper right', fontsize=8); w.canvas.draw()
        except Exception as e: print(f"[EDA] Inspection Error: {e}")

    # --- UTILS & HELPERS ---

    def _get_tab_filtered_df(self, tab):
        """Helper to get a Master DF slice based on the local tab context selector."""
        if not self.last_report or 'df' not in self.last_report: return None
        df = self.last_report['df']
        if not hasattr(tab, 'cb_inspect_ctx'): return df
        sel = tab.cb_inspect_ctx.currentText()
        return df if "All Contexts" in sel else df[df['context'] == sel]

    def _create_plot_subtab(self, callback, mode="hist") -> QWidget:
        """Factory for visual sub-tabs with standard selectors."""
        w = QWidget(); lay = QVBoxLayout(w); h_ctrl = QHBoxLayout()
        w.cb_ch = QComboBox(); h_ctrl.addWidget(QLabel("CH:")); h_ctrl.addWidget(w.cb_ch)
        w.cb_ch.currentIndexChanged.connect(callback)

        w.cb_inspect_ctx = QComboBox()
        h_ctrl.addWidget(QLabel("Context:")); h_ctrl.addWidget(w.cb_inspect_ctx)
        w.cb_inspect_ctx.currentIndexChanged.connect(callback)

        if mode in ["hist", "mapping"]:
            w.cb_metric = QComboBox(); h_ctrl.addWidget(QLabel("Metric:")); h_ctrl.addWidget(w.cb_metric)
            w.cb_metric.currentIndexChanged.connect(callback)
            if mode == "hist":
                w.chk_segregate = QCheckBox("Segregate"); w.chk_segregate.stateChanged.connect(callback)
                h_ctrl.addWidget(w.chk_segregate)
        elif mode == "scatter":
            w.cb_metric_x = QComboBox(); w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X:")); h_ctrl.addWidget(w.cb_metric_x)
            h_ctrl.addWidget(QLabel("Y:")); h_ctrl.addWidget(w.cb_metric_y)
            w.cb_metric_x.currentIndexChanged.connect(callback); w.cb_metric_y.currentIndexChanged.connect(callback)
        elif mode == "inspect":
            w.sb_trial = QSpinBox(); w.sb_trial.setPrefix("Trial #: "); w.sb_trial.setRange(0, 99999)
            w.sb_trial.valueChanged.connect(callback); h_ctrl.addWidget(w.sb_trial)

        if mode != "inspect":
            w.chk_log = QCheckBox("Log"); w.chk_z = QCheckBox("Z"); w.chk_mad = QCheckBox("MAD")
            for c in [w.chk_log, w.chk_z, w.chk_mad]: c.stateChanged.connect(callback); h_ctrl.addWidget(c)
            
        h_ctrl.addStretch(); lay.addLayout(h_ctrl)
        w.fig = Figure(facecolor=self.c_bg, tight_layout=True); w.canvas = FigureCanvas(w.fig)
        lay.addWidget(w.canvas); return w

    def _style_ax(self, ax):
        """Theme styling for Matplotlib axes."""
        ax.set_facecolor(self.c_ax)
        ax.grid(True, ls='--', alpha=0.1, color=self.c_text)
        ax.tick_params(colors=self.c_text, labelsize=9)
        for s in ax.spines.values(): s.set_color(self.c_spine); s.set_linewidth(0.5)
        ax.xaxis.label.set_color(self.c_text); ax.yaxis.label.set_color(self.c_text)
        ax.title.set_color(self.c_text); ax.title.set_weight('bold'); sns.despine(ax=ax)

    def _update_inspection_range(self):
        """Syncs the SpinBox maximum to the selected context's trial count."""
        if not self.last_report: return
        ctx = self.tab_inspect.cb_inspect_ctx.currentText()
        insp = self.last_report.get('inspection_data', {})
        if ctx in insp and insp[ctx].get('raw_traces'):
            first_ch = list(insp[ctx]['raw_traces'].keys())[0]
            self.tab_inspect.sb_trial.setRange(0, max(0, len(insp[ctx]['raw_traces'][first_ch]) - 1))

    def _add_to_history(self, report, params, idx):
        self.history_data.append({'name': f"{self.combo_analyzer.currentText()} #{len(self.history_data)+1}",
                                  'report': report, 'params': params, 'analyzer_idx': idx})
        self.list_history.addItem(self.history_data[-1]['name'])

    def _load_from_history(self, item):
        hist = self.history_data[self.list_history.row(item)]
        self.combo_analyzer.blockSignals(True); self.combo_analyzer.setCurrentIndex(hist['analyzer_idx']); self.combo_analyzer.blockSignals(False)
        self.on_analyzer_changed()
        for k, v in hist['params'].items():
            if k in self.inputs: self.inputs[k].setText(str(v))
        self.last_report = hist['report']; self.show_results(self.last_report); self.setCurrentIndex(1)

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0: self.history_data.pop(row); self.list_history.takeItem(row)

    def _restore_cursor(self):
        while QApplication.overrideCursor(): QApplication.restoreOverrideCursor()
        
        
        
        
import ast
import inspect
import numpy as np
import pandas as pd
import mne
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QPushButton, QScrollArea, QLineEdit, 
                             QFormLayout, QTextEdit, QFrame, QTabWidget, 
                             QListWidget, QApplication, QCheckBox, QSpinBox, QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6 import sip 
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.core.registry import Registry
from src.utils.imports import import_class
from src.interface.widgets.helpers import CheckableComboBox

class EDATab(QTabWidget):
    """
    Advanced Exploratory Data Analysis (EDA) Tab.
    
    Architecture:
    - Stacked UI: Pre-builds configuration panels for every analyzer to prevent crashes.
    - Persistent State: Parameters are kept in memory even if the user switches subjects.
    - Master DataFrame: Merges all context data into a single table for fast Seaborn plotting.
    """
    
    TRACE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
                    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    
    LINE_COLORS = ["#A50F15", "#252525", "#54278F", "#006D2C", "#F0027F", 
                   "#BF812D", "#08306B", "#01665E", "#B35806", "#000000"]

    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # --- 1. STATE INITIALIZATION ---
        self.analyzer_bundles = {}   # Map: {AnalyzerName: {widgets, inputs, selectors}}
        self.history_data = []       # List of dictionaries for history entries
        self.last_report = None      # Stores the currently active Master Report
        self.current_main_param = "data"
        
        self._initialize_aesthetics()
        self._init_ui()
        
        # --- 2. GENERATE ALL ANALYZER UIs ONCE ---
        self._prebuild_analyzer_uis()

    def _initialize_aesthetics(self):
        """Sets up the visual theme for Seaborn and Matplotlib."""
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")
        self.event_colors = sns.color_palette("husl", 12) 
        sns.set_context("paper", font_scale=1.1)

    # --- UI CONSTRUCTION ---

    def _init_ui(self):
        """Builds the main Configuration and Results containers."""
        self.tab_config = QWidget()
        self._build_config_ui()
        
        self.tab_results = QWidget()
        self._build_results_ui()

        self.addTab(self.tab_config, "EDA Configuration")
        self.addTab(self.tab_results, "Diagnostic Results")

    def _build_config_ui(self):
        """Layout: [ History Sidebar | Parameters Setup ]"""
        layout = QHBoxLayout(self.tab_config)

        # 1. Left Sidebar: History
        layout.addWidget(self._create_history_panel())

        # 2. Right Panel: Dynamic Setup
        params_panel = QWidget()
        v_params = QVBoxLayout(params_panel)
        
        # Fixed Form for the global selectors
        self.f_selectors = QFormLayout()
        self._setup_fixed_selectors()
        v_params.addLayout(self.f_selectors)

        # The Stacked Widget (The Dynamic Drawer)
        # This holds the extra signal combos and param fields for every analyzer
        v_params.addWidget(QLabel("<b>Analyzer Configuration:</b>"))
        self.config_stack = QStackedWidget()
        
        # We wrap the stack in a scroll area just in case
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.config_stack)
        v_params.addWidget(scroll)

        # Run Button
        self.btn_run = QPushButton("RUN ANALYTICS")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_analysis)
        v_params.addWidget(self.btn_run)
        
        layout.addWidget(params_panel)

    def _setup_fixed_selectors(self):
        """Standard top-level selectors that never change."""
        # A. Target Analyzer
        self.combo_analyzer = QComboBox()
        available = Registry.discover_eda_analyzers()
        for name, path in available.items():
            self.combo_analyzer.addItem(name, userData=path)
        self.combo_analyzer.currentIndexChanged.connect(self.on_analyzer_changed)

        # B. Common Target Signal (Maps to the first arg of analyze)
        self.lbl_main_signal = QLabel("<b>Target Signal(s):</b>")
        self.combo_focus = CheckableComboBox()
        self.combo_focus.setPlaceholderText("Select data context...")

        self.f_selectors.addRow("<b>Target Analyzer:</b>", self.combo_analyzer)
        self.f_selectors.addRow(self.lbl_main_signal, self.combo_focus)

    def _create_history_panel(self) -> QFrame:
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

    # --- THE PRE-BUILD ENGINE ---

    def _prebuild_analyzer_uis(self):
        """Creates a hidden UI panel for every analyzer found in the Registry."""
        available = Registry.discover_eda_analyzers()
        
        for name, path in available.items():
            # Create a dedicated container widget for this specific analyzer
            bundle_widget = QWidget()
            layout = QVBoxLayout(bundle_widget)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # --- PART A: EXTRA SIGNAL SELECTORS ---
            extra_sigs_form = QFormLayout()
            signal_selectors = {}
            
            Cls = import_class(name, path)
            analyze_sig = inspect.signature(Cls.analyze)
            params = [p for p in analyze_sig.parameters.values() if p.name != 'self']
            
            # The first parameter is shared via combo_focus, we handle others:
            if len(params) > 1:
                for extra in params[1:]:
                    label_text = f"<b>{extra.name.replace('_',' ').title()}:</b>"
                    combo = CheckableComboBox()
                    combo.setPlaceholderText(f"Select {extra.name}...")
                    extra_sigs_form.addRow(label_text, combo)
                    signal_selectors[extra.name] = combo
            
            layout.addLayout(extra_sigs_form)

            # --- PART B: ARGUMENT INPUTS (__init__) ---
            layout.addWidget(QLabel("<b>Analyzer Parameters:</b>"))
            params_form = QFormLayout()
            inputs = {}
            defaults = Registry.get_analyzer_params(name, path)
            
            for arg, val in defaults.items():
                if arg in ['self', 'kwargs', 'args']: continue
                le = QLineEdit(str(val))
                params_form.addRow(QLabel(f"<i>{arg}</i>:"), le)
                inputs[arg] = le
            
            layout.addLayout(params_form)
            layout.addStretch() # Layout stability

            # Store the bundle
            self.analyzer_bundles[name] = {
                "widget": bundle_widget,
                "inputs": inputs,
                "selectors": signal_selectors,
                "main_param_name": params[0].name if params else "data"
            }
            
            # Add to the visual stack
            self.config_stack.addWidget(bundle_widget)
            
        # Set initial visibility
        self.on_analyzer_changed()

    # --- LOGIC & SWITCHING ---

    def on_analyzer_changed(self):
        """Instant switch between pre-built UIs without deleting anything."""
        name = self.combo_analyzer.currentText()
        if name in self.analyzer_bundles:
            bundle = self.analyzer_bundles[name]
            # Flip the stack to the pre-built widget
            self.config_stack.setCurrentWidget(bundle['widget'])
            # Update the label of the shared Focus Signal selector
            self.current_main_param = bundle['main_param_name']
            label_text = self.current_main_param.replace('_',' ').title()
            self.lbl_main_signal.setText(f"<b>{label_text}(s):</b>")
            
            # Trigger a refresh of the signal names in the newly shown combo boxes
            self._populate_all_signal_selectors()

    def refresh(self):
        """Syncs signal lists across the entire pre-built stack."""
        if sip.isdeleted(self) or not self.session.workspace: return
        self._populate_all_signal_selectors()
        if self.session.eda_report:
            self.show_results(self.session.eda_report)

    def _populate_all_signal_selectors(self):
        """Ensures every combo box in every bundle has the current workspace keys."""
        all_keys = list(self.session.workspace.keys())

        # 1. Update shared Focus Combo
        self.combo_focus.blockSignals(True)
        self.combo_focus.clear()
        for k in all_keys: self.combo_focus.addItem(k)
        self.combo_focus.blockSignals(False)

        # 2. Update every extra signal combo in the hidden bundles
        for name, bundle in self.analyzer_bundles.items():
            for combo in bundle['selectors'].values():
                combo.blockSignals(True)
                combo.clear()
                for k in all_keys: combo.addItem(k)
                combo.blockSignals(False)

    # --- EXECUTION ENGINE ---

    def run_analysis(self):
        """Reads from the active stack bundle and executes the analysis."""
        name = self.combo_analyzer.currentText()
        bundle = self.analyzer_bundles[name]
        
        focus_keys = self.combo_focus.get_checked_items()
        if not focus_keys: return

        # Clear focus to prevent 'ghost' input crashes
        if QApplication.focusObject(): QApplication.focusObject().clearFocus()
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            # 1. Gather Params
            kwargs_init = {k: ast.literal_eval(v.text()) if v.text().strip() else "" 
                           for k, v in bundle['inputs'].items()}
            
            # 2. Gather Signal Objects
            extra_sigs = {}
            for arg, combo in bundle['selectors'].items():
                sel_keys = combo.get_checked_items()
                objs = [self.session.workspace[k] for k in sel_keys]
                extra_sigs[arg] = objs[0] if len(objs) == 1 else (objs if objs else None)

            # 3. Instantiate and Execute via Merge Engine
            Cls = import_class(name, self.combo_analyzer.currentData())
            analyzer = Cls(**kwargs_init)
            
            report = self._execute_multi_context(analyzer, focus_keys, extra_sigs)
            
            # 4. Finalize
            self.session.eda_report = report
            # Save raw strings for UI restoration later
            raw_params = {k: v.text() for k, v in bundle['inputs'].items()}
            self._add_to_history(report, raw_params, self.combo_analyzer.currentIndex())
            
            self.show_results(report)
            self.setCurrentIndex(1)
        except Exception as e:
            print(f"[EDA] Analysis Execution Error: {e}")
        finally:
            self._restore_cursor()

    def _execute_multi_context(self, analyzer, focus_keys, extra_sigs) -> dict:
        """Merges contexts into a Tidy Master DataFrame for Comparative Analysis."""
        all_frames, summaries, inspection_map = [], [], {}
        global_counter = 0

        for ctx_key in focus_keys:
            # Mapping: first arg of analyze() gets the focus signal
            call_kwargs = {self.current_main_param: self.session.workspace[ctx_key]}
            call_kwargs.update(extra_sigs)
            
            report = analyzer.analyze(**call_kwargs)
            summaries.append(f"[{ctx_key}]\n{report['summary']}")

            # Tidy unrolling into Pandas
            metrics = report['metrics']
            cues = report['artifacts'].get('event_types', [])
            chans = list(metrics[list(metrics.keys())[0]]['values'].keys())
            
            records = []
            for i, cue_id in enumerate(cues):
                is_bad = i in report['artifacts']['bad_indices']
                for ch in chans:
                    bad_flag = False
                    if is_bad:
                        pos = report['artifacts']['bad_indices'].index(i)
                        bad_flag = any(ch.lower() in t.lower() for t in report['artifacts']['bad_tags'][pos])
                    
                    row = {
                        'context': ctx_key, 'cue': f"{ctx_key}_{cue_id}", 'channel': ch,
                        'trial_idx': i, 'global_idx': global_counter + i, 'is_flagged': bad_flag,
                        **{m: m_data['values'][ch][i] for m, m_data in metrics.items()}
                    }
                    records.append(row)
            
            all_frames.append(pd.DataFrame(records))
            inspection_map[ctx_key] = report.get('inspection_data', {})
            global_counter += len(cues)

        return {
            "summary": "\n---\n".join(summaries),
            "df": pd.concat(all_frames, ignore_index=True),
            "visuals": report.get("visuals", ["hist", "scatter", "mapping"]),
            "artifacts": report['artifacts'],
            "inspection_data": inspection_map
        }

    # --- VISUALIZATION & RESULTS ---

    def _build_results_ui(self):
        """Constructs the results container with Seaborn plots."""
        layout = QVBoxLayout(self.tab_results)
        self.res_subtabs = QTabWidget()
        
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        self.tab_hist = self._create_plot_subtab(self.update_hist_canvas, mode="hist")
        self.tab_scatter = self._create_plot_subtab(self.update_scatter_canvas, mode="scatter")
        self.tab_mapping = self._create_plot_subtab(self.update_mapping_canvas, mode="mapping")
        self.tab_inspect = self._create_plot_subtab(self.update_inspect_canvas, mode="inspect")

        self.res_subtabs.addTab(self.txt_report, "Text Summary")
        self.res_subtabs.addTab(self.tab_hist, "Histograms")
        self.res_subtabs.addTab(self.tab_scatter, "Scatters")
        self.res_subtabs.addTab(self.tab_mapping, "Sequence Mapping")
        self.res_subtabs.addTab(self.tab_inspect, "Signal Inspection")
        layout.addWidget(self.res_subtabs)

    def show_results(self, report):
        """Populates the results sub-tabs based on the Master DF."""
        if not report or 'df' not in report: return
        self.last_report = report
        self.txt_report.setText(report.get('summary', ""))
        
        supported = report.get("visuals", ["hist", "scatter", "mapping", "inspect"])
        tabs_to_update = []
        mapping = {"hist": (1, self.tab_hist), "scatter": (2, self.tab_scatter), 
                   "mapping": (3, self.tab_mapping), "inspect": (4, self.tab_inspect)}

        for tag, (idx, tab_obj) in mapping.items():
            visible = tag in supported
            self.res_subtabs.setTabVisible(idx, visible)
            if visible: tabs_to_update.append(tab_obj)

        # Extraction from DF
        df = report['df']
        chans = sorted(df['channel'].unique().tolist())
        meta = ['context', 'cue', 'channel', 'trial_idx', 'is_flagged', 'global_idx']
        metrics = [c for c in df.columns if c not in meta and not c.endswith('_unit')]

        for tab in tabs_to_update:
            self._update_tab_selectors(tab, metrics, chans)

        if "hist" in supported:
            self.res_subtabs.setCurrentIndex(1)
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self.update_hist_canvas)

    def _update_tab_selectors(self, tab, metrics, chans):
        """Populates the local combo boxes in each sub-tab."""
        combos = [(getattr(tab, 'cb_ch', None), chans, True),
                  (getattr(tab, 'cb_metric', None), metrics, False),
                  (getattr(tab, 'cb_metric_x', None), metrics, False),
                  (getattr(tab, 'cb_metric_y', None), metrics, False)]

        for cb, items, overlay in combos:
            if cb:
                cb.blockSignals(True); cb.clear()
                if overlay: cb.addItem("All Channels (Overlay)")
                cb.addItems(items); cb.setCurrentIndex(0); cb.blockSignals(False)

        if hasattr(tab, 'cb_inspect_ctx'):
            cb_ctx = tab.cb_inspect_ctx
            cb_ctx.blockSignals(True); cb_ctx.clear()
            if tab != self.tab_inspect: cb_ctx.addItem("All Contexts (Comparison)")
            cb_ctx.addItems(sorted(self.last_report['df']['context'].unique().tolist()))
            cb_ctx.setCurrentIndex(0); cb_ctx.blockSignals(False)
            
        if tab == self.tab_inspect: self._update_inspection_range()

    # --- CANVAS PLOTTERS ---

    def update_hist_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_hist
        df = self._get_tab_filtered_df(w)
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return
        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]
        
        df = df.copy(); df[metric] = self._apply_transforms(df[metric].values, w)
        hue_col = 'cue' if (hasattr(w, 'chk_segregate') and w.chk_segregate.isChecked()) else ('context' if "All Contexts" in w.cb_inspect_ctx.currentText() else None)

        try:
            sns.histplot(data=df, x=metric, hue=hue_col, palette=self.event_colors, 
                         ax=ax, kde=True, element="step", common_norm=False, stat='probability')
            ax.set_title(f"Distribution: {metric} ({ch})")
        except: pass
        w.canvas.draw()

    def update_scatter_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_scatter
        df = self._get_tab_filtered_df(w)
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        mx, my, ch = w.cb_metric_x.currentText(), w.cb_metric_y.currentText(), w.cb_ch.currentText()
        if not mx or not my or not ch: return
        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]

        df = df.copy()
        df[mx], df[my] = self._apply_transforms(df[mx].values, w), self._apply_transforms(df[my].values, w)
        hue = 'context' if "All Contexts" in w.cb_inspect_ctx.currentText() else None
        
        sns.scatterplot(data=df, x=mx, y=my, hue=hue, palette=self.event_colors, ax=ax, alpha=0.5)
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty: ax.scatter(flagged[mx], flagged[my], color='red', marker='x', s=80, label='Flagged')
        ax.set_title(f"Correlation: {mx} vs {my}")
        w.canvas.draw()

    def update_mapping_canvas(self):
        if not self.last_report or 'df' not in self.last_report: return
        w = self.tab_mapping
        df = self._get_tab_filtered_df(w)
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if df is None or df.empty: return

        metric, ch = w.cb_metric.currentText(), w.cb_ch.currentText()
        if not metric or not ch: return
        if ch != "All Channels (Overlay)": df = df[df['channel'] == ch]
        
        df = df.copy(); df[metric] = self._apply_transforms(df[metric].values, w)
        hue = 'context' if "All Contexts" in w.cb_inspect_ctx.currentText() else None
        
        sns.scatterplot(data=df, x='trial_idx', y=metric, hue=hue, palette=self.event_colors, ax=ax, s=40)
        flagged = df[df['is_flagged'] == True]
        if not flagged.empty: ax.scatter(flagged['trial_idx'], flagged[metric], color='red', marker='x', s=100)
        ax.set_title(f"Sequence: {metric} ({ch})")
        w.canvas.draw()

    def update_inspect_canvas(self):
        if not self.last_report or 'inspection_data' not in self.last_report: return
        w = self.tab_inspect
        self._update_inspection_range()
        ctx_name, ch, tid = w.cb_inspect_ctx.currentText(), w.cb_ch.currentText(), w.sb_trial.value()
        
        insp_map = self.last_report['inspection_data']
        if ctx_name not in insp_map: return
        ctx_data = insp_map[ctx_name]
        
        w.fig.clear(); ax = w.fig.add_subplot(111); self._style_ax(ax)
        if ch == "All Channels (Overlay)" or ch not in ctx_data.get('raw_traces', {}):
            ax.text(0.5, 0.5, "Select specific channel", ha='center', color='gray'); w.canvas.draw(); return

        try:
            t = np.linspace(self.last_report['artifacts'].get('tmin', 0), 
                            self.last_report['artifacts'].get('tmax', 1), 
                            len(ctx_data['raw_traces'][ch][tid]))
            
            idx_t, idx_l = 0, 0
            for key, data in ctx_data.items():
                if key.endswith("_traces"):
                    ax.plot(t, np.array(data[ch][tid]), color=self.TRACE_COLORS[idx_t % 10], label=key.replace("_traces", ""))
                    idx_t += 1
                elif key.endswith("_lines"):
                    for i, val in enumerate(data[ch]):
                        ax.axhline(y=val, color=self.LINE_COLORS[idx_l % 10], ls='--', label=f"{key}_{i}")
                        idx_l += 1
            
            bad = self.last_report['df'][(self.last_report['df']['context'] == ctx_name) & 
                                        (self.last_report['df']['trial_idx'] == tid) & 
                                        (self.last_report['df']['channel'] == ch)]['is_flagged'].any()
            ax.set_title(f"{'[BAD]' if bad else '[OK]'} {ctx_name} | {ch} | Trial #{tid}")
            ax.legend(frameon=False, loc='upper right', fontsize=8); w.canvas.draw()
        except Exception as e: print(f"[EDA] Inspection Error: {e}")

    # --- HELPERS ---

    def _get_tab_filtered_df(self, tab):
        if not self.last_report or 'df' not in self.last_report: return None
        df = self.last_report['df']
        if not hasattr(tab, 'cb_inspect_ctx'): return df
        sel = tab.cb_inspect_ctx.currentText()
        return df if "All Contexts" in sel else df[df['context'] == sel]

    def _create_plot_subtab(self, callback, mode="hist") -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); h_ctrl = QHBoxLayout()
        w.cb_ch = QComboBox(); h_ctrl.addWidget(QLabel("CH:")); h_ctrl.addWidget(w.cb_ch)
        w.cb_ch.currentIndexChanged.connect(callback)
        w.cb_inspect_ctx = QComboBox(); h_ctrl.addWidget(QLabel("Context:")); h_ctrl.addWidget(w.cb_inspect_ctx)
        w.cb_inspect_ctx.currentIndexChanged.connect(callback)

        if mode in ["hist", "mapping"]:
            w.cb_metric = QComboBox(); h_ctrl.addWidget(QLabel("Metric:")); h_ctrl.addWidget(w.cb_metric)
            w.cb_metric.currentIndexChanged.connect(callback)
            if mode == "hist":
                w.chk_segregate = QCheckBox("Segregate"); w.chk_segregate.stateChanged.connect(callback)
                h_ctrl.addWidget(w.chk_segregate)
        elif mode == "scatter":
            w.cb_metric_x = QComboBox(); w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X:")); h_ctrl.addWidget(w.cb_metric_x); h_ctrl.addWidget(QLabel("Y:")); h_ctrl.addWidget(w.cb_metric_y)
            w.cb_metric_x.currentIndexChanged.connect(callback); w.cb_metric_y.currentIndexChanged.connect(callback)
        elif mode == "inspect":
            w.sb_trial = QSpinBox(); w.sb_trial.setPrefix("Trial #: "); w.sb_trial.setRange(0, 99999)
            w.sb_trial.valueChanged.connect(callback); h_ctrl.addWidget(w.sb_trial)

        if mode != "inspect":
            w.chk_log = QCheckBox("Log"); w.chk_z = QCheckBox("Z"); w.chk_mad = QCheckBox("MAD")
            for c in [w.chk_log, w.chk_z, w.chk_mad]: c.stateChanged.connect(callback); h_ctrl.addWidget(c)
            
        h_ctrl.addStretch(); lay.addLayout(h_ctrl)
        w.fig = Figure(facecolor=self.c_bg, tight_layout=True); w.canvas = FigureCanvas(w.fig)
        lay.addWidget(w.canvas); return w

    def _style_ax(self, ax):
        ax.set_facecolor(self.c_ax); ax.grid(True, ls='--', alpha=0.1, color=self.c_text)
        ax.tick_params(colors=self.c_text, labelsize=9)
        for s in ax.spines.values(): s.set_color(self.c_spine); s.set_linewidth(0.5)
        ax.xaxis.label.set_color(self.c_text); ax.yaxis.label.set_color(self.c_text); ax.title.set_color(self.c_text); ax.title.set_weight('bold'); sns.despine(ax=ax)

    def _apply_transforms(self, data, sub_tab):
        res = np.array(data, dtype=float)
        if hasattr(sub_tab, 'chk_log') and sub_tab.chk_log.isChecked(): res = np.log10(np.maximum(res, 1e-12))
        if hasattr(sub_tab, 'chk_z') and sub_tab.chk_z.isChecked(): res = (res - np.nanmean(res)) / (np.nanstd(res) + 1e-9)
        elif hasattr(sub_tab, 'chk_mad') and sub_tab.chk_mad.isChecked():
            med = np.nanmedian(res); mad = np.nanmedian(np.abs(res - med))
            res = (res - med) / (mad * 1.4826 + 1e-9)
        return res

    def _update_inspection_range(self):
        if not self.last_report: return
        ctx = self.tab_inspect.cb_inspect_ctx.currentText()
        insp = self.last_report.get('inspection_data', {})
        if ctx in insp and insp[ctx].get('raw_traces'):
            first_ch = list(insp[ctx]['raw_traces'].keys())[0]
            self.tab_inspect.sb_trial.setRange(0, max(0, len(insp[ctx]['raw_traces'][first_ch]) - 1))

    def _add_to_history(self, report, params, idx):
        self.history_data.append({'name': f"{self.combo_analyzer.currentText()} #{len(self.history_data)+1}", 'report': report, 'params': params, 'analyzer_idx': idx})
        self.list_history.addItem(self.history_data[-1]['name'])

    def _load_from_history(self, item):
        hist = self.history_data[self.list_history.row(item)]
        self.combo_analyzer.blockSignals(True); self.combo_analyzer.setCurrentIndex(hist['analyzer_idx']); self.combo_analyzer.blockSignals(False)
        self.on_analyzer_changed()
        bundle = self.analyzer_bundles[self.combo_analyzer.currentText()]
        for k, v in hist['params'].items():
            if k in bundle['inputs']: bundle['inputs'][k].setText(str(v))
        self.last_report = hist['report']; self.show_results(self.last_report); self.setCurrentIndex(1)

    def _remove_selected_history(self):
        row = self.list_history.currentRow()
        if row >= 0: self.history_data.pop(row); self.list_history.takeItem(row)

    def _restore_cursor(self):
        while QApplication.overrideCursor(): QApplication.restoreOverrideCursor()