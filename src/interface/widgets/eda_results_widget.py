import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTextEdit, 
                             QLabel, QComboBox, QSpinBox, QHBoxLayout, QCheckBox, QPushButton)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.eda.eda_report import EDAReport
from src.interface.widgets.helpers import CheckableComboBox

class EDAResultsWidget(QWidget):
    def __init__(self, theme_kwargs=None, parent=None):
        super().__init__(parent)
        self.reports = [] 
        self.df = None
        
        # --- THEME CONFIGURATION ---
        tk = theme_kwargs or {}
        self.c_bg = tk.get("figure.facecolor", "#ffffff")
        self.c_fg = tk.get("text.color", "#000000")
        self.c_ax = tk.get("axes.facecolor", "#f0f0f0")
        self.c_grid = tk.get("grid.color", "#d3d3d3")
        
        self.TRACE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", 
                             "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
        self.LINE_COLORS = ["#A50F15", "#252525", "#08519C", "#006D2C"]
        
        self._build_ui()

    def _build_ui(self):
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # 1. Summary
        self.tab_summary = QWidget()
        self.text_summary = QTextEdit()
        self.text_summary.setReadOnly(True)
        QVBoxLayout(self.tab_summary).addWidget(self.text_summary)
        self.tabs.addTab(self.tab_summary, "Text Summary")

        # 2. Dynamic Plot Tabs
        self.tab_hist = self._create_plot_subtab(self._draw_histograms, mode="hist")
        self.tab_scatter = self._create_plot_subtab(self._draw_scatter, mode="scatter")
        self.tab_boxplot = self._create_plot_subtab(self._draw_boxplot, mode="boxplot")
        self.tab_mapping = self._create_plot_subtab(self._draw_mapping, mode="mapping")
        self.tab_inspect = self._create_plot_subtab(self._draw_inspection, mode="inspect")

    def _create_plot_subtab(self, callback, mode="hist") -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w); h_ctrl = QHBoxLayout()
        
        # Assign Widget Type Based on Tab Mode
        if mode == "inspect":
            w.cb_ch = QComboBox()
            w.cb_data = QComboBox()
            w.cb_ch.currentIndexChanged.connect(callback)
            w.cb_data.currentIndexChanged.connect(callback)
        else:
            w.cb_ch = CheckableComboBox()
            w.cb_data = CheckableComboBox()

        h_ctrl.addWidget(QLabel("CH:"))
        h_ctrl.addWidget(w.cb_ch)
        h_ctrl.addWidget(QLabel("Data:"))
        h_ctrl.addWidget(w.cb_data)

        # Refresh button only for multi-select tabs to prevent lag
        if mode != "inspect":
            w.btn_update = QPushButton("Refresh Plot")
            w.btn_update.clicked.connect(callback)
            h_ctrl.addWidget(w.btn_update)

        # Mode-specific metrics
        if mode in ["hist", "boxplot", "mapping"]:
            w.cb_metric = QComboBox()
            w.cb_metric.currentIndexChanged.connect(callback)
            h_ctrl.addWidget(QLabel("Metric:")); h_ctrl.addWidget(w.cb_metric)
            
        elif mode == "scatter":
            w.cb_metric_x = QComboBox(); w.cb_metric_y = QComboBox()
            h_ctrl.addWidget(QLabel("X:")); h_ctrl.addWidget(w.cb_metric_x)
            h_ctrl.addWidget(QLabel("Y:")); h_ctrl.addWidget(w.cb_metric_y)
            w.cb_metric_x.currentIndexChanged.connect(callback)
            w.cb_metric_y.currentIndexChanged.connect(callback)
            
        elif mode == "inspect":
            w.sb_trial = QSpinBox()
            w.sb_trial.setPrefix("Trial #: "); w.sb_trial.setRange(0, 99999)
            w.sb_trial.valueChanged.connect(callback); h_ctrl.addWidget(w.sb_trial)

        if mode != "inspect":
            w.chk_log = QCheckBox("Log")
            w.chk_z = QCheckBox("Z")
            w.chk_mad = QCheckBox("MAD")
            for c in [w.chk_log, w.chk_z, w.chk_mad]: 
                c.stateChanged.connect(callback)
                h_ctrl.addWidget(c)
            
        h_ctrl.addStretch(); lay.addLayout(h_ctrl)
        w.fig = Figure(facecolor=self.c_bg, tight_layout=True); w.canvas = FigureCanvas(w.fig)
        lay.addWidget(w.canvas); return w

    def display_report(self, reports: Union[EDAReport, List[EDAReport]]):
        if not isinstance(reports, list):
            reports = [reports]
        self.reports = reports
        
        full_summary = "\n\n--- NEXT CONTEXT ---\n\n".join([r.summary for r in reports])
        self.text_summary.setText(full_summary)
        
        visuals = reports[0].visuals if reports else []
        self._manage_tabs(visuals)

        self.df = self._build_dataframe(reports)
        if self.df.empty: return

        channels_list = list(self.df["Channel"].unique())
        data_list = list(self.df["Tag"].unique())
        metrics_list = list(self.df["Metric"].unique())

        self._block_all_signals(True)
        
        # Multi-Select Tabs: Add "All" option
        for w in [self.tab_hist, self.tab_scatter, self.tab_boxplot, self.tab_mapping]:
            w.cb_ch.clear(); w.cb_ch.addItems(["All"] + channels_list)
            w.cb_data.clear(); w.cb_data.addItems(["All"] + data_list)
            
            # Check ONLY "All" by default to keep the UI clean
            w.cb_ch.model().item(0).setCheckState(Qt.CheckState.Checked)
            w.cb_data.model().item(0).setCheckState(Qt.CheckState.Checked)

        # Inspect Tab: Standard single-select (No "All")
        self.tab_inspect.cb_ch.clear(); self.tab_inspect.cb_ch.addItems(channels_list)
        self.tab_inspect.cb_data.clear(); self.tab_inspect.cb_data.addItems(data_list)

        if "hist" in visuals or "mapping" in visuals or "boxplot" in visuals:
            self.tab_hist.cb_metric.clear(); self.tab_hist.cb_metric.addItems(metrics_list)
            self.tab_boxplot.cb_metric.clear(); self.tab_boxplot.cb_metric.addItems(metrics_list)
            self.tab_mapping.cb_metric.clear(); self.tab_mapping.cb_metric.addItems(metrics_list)
        
        if "scatter" in visuals:
            self.tab_scatter.cb_metric_x.clear(); self.tab_scatter.cb_metric_x.addItems(metrics_list)
            self.tab_scatter.cb_metric_y.clear(); self.tab_scatter.cb_metric_y.addItems(metrics_list)

        self._block_all_signals(False)

        # Trigger initial plots
        if "hist" in visuals: self._draw_histograms()
        if "scatter" in visuals: self._draw_scatter()
        if "boxplot" in visuals: self._draw_boxplot()
        if "mapping" in visuals: self._draw_mapping()
        if "inspect" in visuals: self._draw_inspection()

    def _block_all_signals(self, block: bool):
        for w in [self.tab_hist, self.tab_scatter, self.tab_boxplot, self.tab_mapping, self.tab_inspect]:
            w.cb_ch.blockSignals(block); w.cb_data.blockSignals(block)
            if hasattr(w, 'cb_metric'): w.cb_metric.blockSignals(block)
            if hasattr(w, 'cb_metric_x'): w.cb_metric_x.blockSignals(block)
            if hasattr(w, 'cb_metric_y'): w.cb_metric_y.blockSignals(block)

    def _manage_tabs(self, visuals):
        for i in range(self.tabs.count() - 1, 0, -1):
            self.tabs.removeTab(i)
        if "hist" in visuals: self.tabs.addTab(self.tab_hist, "Histograms")
        if "scatter" in visuals: self.tabs.addTab(self.tab_scatter, "Scatter Pairs")
        if "mapping" in visuals: 
            self.tabs.addTab(self.tab_boxplot, "Boxplots")
            self.tabs.addTab(self.tab_mapping, "Mapping (QC)")
        if "inspect" in visuals: self.tabs.addTab(self.tab_inspect, "Inspect Data")

    def _build_dataframe(self, reports: List[EDAReport]) -> pd.DataFrame:
        records = []
        for report in reports:
            ctx = report.metadata.get("context", "Context")
            events = report.metadata.get("event_types", [])
            bad_idx = report.artifacts.bad_indices if report.artifacts else []
            
            for metric_name, metric_obj in report.metrics.items():
                for ch, values in metric_obj.values.items():
                    for trial_idx, val in enumerate(values):
                        event_tag = str(events[trial_idx]) if trial_idx < len(events) else "Unknown"
                        tag = f"{ctx}_{event_tag}" 
                        is_bad = trial_idx in bad_idx
                        
                        records.append({
                            "Context": ctx,
                            "Metric": metric_name,
                            "Channel": ch,
                            "Trial": trial_idx,
                            "Value": val,
                            "Tag": tag,
                            "Is_Bad": is_bad,
                            "Unit": metric_obj.unit
                        })
        return pd.DataFrame(records)

    def _apply_math_transforms(self, df_slice, tab_widget):
        vals = df_slice["Value"].copy()
        if tab_widget.chk_log.isChecked(): vals = np.log1p(np.abs(vals))
        if tab_widget.chk_z.isChecked(): vals = (vals - vals.mean()) / (vals.std() + 1e-9)
        if tab_widget.chk_mad.isChecked():
            median = vals.median(); mad = np.median(np.abs(vals - median))
            vals = (vals - median) / (mad * 1.4826 + 1e-9)
        df_slice.loc[:, "Value"] = vals
        return df_slice

    def _apply_theme(self, fig, ax):
        fig.patch.set_facecolor(self.c_bg); ax.set_facecolor(self.c_ax)
        ax.tick_params(colors=self.c_fg)
        ax.xaxis.label.set_color(self.c_fg); ax.yaxis.label.set_color(self.c_fg)
        ax.title.set_color(self.c_fg)
        for spine in ax.spines.values(): spine.set_color(self.c_grid)
        ax.grid(True, color=self.c_grid, linestyle='--', alpha=0.5)

    def _get_filtered_df(self, w, metric_col="Metric"):
        """Extract multi-select UI values and filter the dataframe, handling the 'All' logic."""
        if self.df is None or self.df.empty: return pd.DataFrame()
        
        chs = w.cb_ch.get_checked_items() if hasattr(w.cb_ch, 'get_checked_items') else [w.cb_ch.currentText()]
        data_tags = w.cb_data.get_checked_items() if hasattr(w.cb_data, 'get_checked_items') else [w.cb_data.currentText()]
        metric = w.cb_metric.currentText() if hasattr(w, 'cb_metric') else None
        
        # Interpret "All" as selecting everything
        if "All" in chs or not chs:
            chs = self.df["Channel"].unique()
        if "All" in data_tags or not data_tags:
            data_tags = self.df["Tag"].unique()
            
        mask = self.df["Channel"].isin(chs) & self.df["Tag"].isin(data_tags)
        if metric: mask &= (self.df[metric_col] == metric)
        
        return self.df[mask].copy()

    # ==========================================
    # PLOTTING ROUTINES 
    # ==========================================
    def _draw_histograms(self, *args):
        w = self.tab_hist; w.fig.clear()
        df_plot = self._get_filtered_df(w)
        if df_plot.empty: return
        
        df_plot = self._apply_math_transforms(df_plot, w)
        ax = w.fig.add_subplot(111)
        
        sns.histplot(data=df_plot, x="Value", hue="Tag", kde=True, ax=ax, palette="deep")
        
        ax.set_title(f"Histogram: {w.cb_metric.currentText()}")
        self._apply_theme(w.fig, ax)
        w.canvas.draw()

    def _draw_scatter(self, *args):
        w = self.tab_scatter; w.fig.clear()
        
        chs = w.cb_ch.get_checked_items()
        data_tags = w.cb_data.get_checked_items()
        m_x = w.cb_metric_x.currentText(); m_y = w.cb_metric_y.currentText()
        
        if "All" in chs or not chs: chs = self.df["Channel"].unique()
        if "All" in data_tags or not data_tags: data_tags = self.df["Tag"].unique()
        
        mask = self.df["Channel"].isin(chs) & self.df["Tag"].isin(data_tags)
        df_base = self.df[mask]
        
        df_x = self._apply_math_transforms(df_base[df_base["Metric"] == m_x].copy(), w).rename(columns={"Value": "X"})
        df_y = self._apply_math_transforms(df_base[df_base["Metric"] == m_y].copy(), w).rename(columns={"Value": "Y"})
        
        df_plot = pd.merge(df_x[["Trial", "Context", "Channel", "Tag", "X"]], 
                           df_y[["Trial", "Context", "Channel", "Tag", "Y"]], 
                           on=["Trial", "Context", "Channel", "Tag"])
        
        if df_plot.empty: return
        ax = w.fig.add_subplot(111)
        sns.scatterplot(data=df_plot, x="X", y="Y", hue="Tag", style="Channel", ax=ax, palette="deep", alpha=0.8)
        
        ax.set_xlabel(m_x); ax.set_ylabel(m_y)
        ax.set_title(f"Scatter: {m_x} vs {m_y}")
        self._apply_theme(w.fig, ax)
        w.canvas.draw()

    def _draw_boxplot(self, *args):
        w = self.tab_boxplot; w.fig.clear()
        df_plot = self._get_filtered_df(w)
        if df_plot.empty: return
        
        df_plot = self._apply_math_transforms(df_plot, w)
        ax = w.fig.add_subplot(111)
        
        sns.boxplot(data=df_plot, x="Channel", y="Value", hue="Tag", ax=ax, showfliers=False, palette="deep")
        sns.stripplot(data=df_plot, x="Channel", y="Value", hue="Tag", ax=ax, dodge=True, alpha=0.6, palette="dark")
        
        ax.set_title(f"Distributions: {w.cb_metric.currentText()}")
        self._apply_theme(w.fig, ax)
        w.canvas.draw()

    def _draw_mapping(self, *args):
        w = self.tab_mapping; w.fig.clear()
        df_plot = self._get_filtered_df(w)
        if df_plot.empty: return
        
        df_plot = self._apply_math_transforms(df_plot, w)
        ax = w.fig.add_subplot(111)
        
        sns.scatterplot(data=df_plot, x="Trial", y="Value", hue="Tag", style="Channel", ax=ax, palette="deep", s=50)
        
        bad_df = df_plot[df_plot["Is_Bad"] == True]
        if not bad_df.empty:
            ax.scatter(bad_df["Trial"], bad_df["Value"], color="red", marker="X", s=150, zorder=5, label="Artifact (Bad)")

        ax.set_title(f"Mapping (QC): {w.cb_metric.currentText()}")
        ax.set_xlabel("Epoch / Trial Index")
        self._apply_theme(w.fig, ax)
        w.canvas.draw()

    def _draw_inspection(self, *args):
        if not self.reports: return
        w = self.tab_inspect; w.fig.clear()
        
        trial = w.sb_trial.value()
        ch = w.cb_ch.currentText()
        selected_data = w.cb_data.currentText()
        
        # 1. Match the selected Data string back to its corresponding report context
        target_report = self.reports[0]
        for rep in self.reports:
            ctx = rep.metadata.get("context", "")
            if selected_data.startswith(ctx):
                target_report = rep
                break

        insp_data = target_report.inspection_data
        if not insp_data: return
            
        ax = w.fig.add_subplot(111)
        x_axis = insp_data.get("x-axis", None)
        trace_idx = 0; line_idx = 0
        
        for key, ch_dict in insp_data.items():
            if key == "x-axis": continue
            if "traces" in key and ch in ch_dict and trial < len(ch_dict[ch]):
                data = ch_dict[ch][trial]
                color = self.TRACE_COLORS[trace_idx % len(self.TRACE_COLORS)]
                label_name = key.replace("_traces", "").replace("_", " ").title()
                if x_axis is not None and len(x_axis) == len(data):
                    ax.plot(x_axis, data, label=label_name, color=color)
                else: ax.plot(data, label=label_name, color=color)
                trace_idx += 1
            elif "lines" in key and ch in ch_dict:
                val = ch_dict[ch]
                if isinstance(val, (list, np.ndarray)) and trial < len(val): val = val[trial]
                color = self.LINE_COLORS[line_idx % len(self.LINE_COLORS)]
                label_name = key.replace("_lines", "").replace("_", " ").title()
                ax.axhline(val, label=label_name, color=color, linestyle="--", linewidth=1.5)
                line_idx += 1

        ax.set_title(f"Inspection: Trial {trial} - Channel {ch} ({selected_data})")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(facecolor=self.c_bg, edgecolor=self.c_grid, labelcolor=self.c_fg)
        self._apply_theme(w.fig, ax)
        w.canvas.draw()