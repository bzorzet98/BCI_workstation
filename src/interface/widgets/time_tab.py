
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollBar, QSpinBox, QCheckBox, QSizePolicy,
                             QTabWidget, QGroupBox, QComboBox)
from PyQt6.QtCore import Qt
from src.interface.widgets.channel_tab import ChannelTab

class TimeDomainTab(QWidget):
    """
    Main Time Series Analysis Container.
    Layout:
    - Persistent Header: Context (Signal) Selector.
    - Sub-Tabs: 1. Channel Selection | 2. Time Plot.
    """
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # Style parameters
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")
        self.event_colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6', '#3498db', '#e67e22']

        self._updating = False
        self.current_start_time = 0.0

        self._init_ui()

    def _init_ui(self):
        """Initializes the layout, moving focus control to the tab corner to save space."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5) # Tight margins

        # ==========================================
        # 1. CREATE THE INTERNAL TAB SYSTEM
        # ==========================================
        self.sub_tabs = QTabWidget()
        
        # Tab 1: Channel Selector
        self.tab_channels = ChannelTab(self.session, self.style_kwargs)
        
        # Tab 2: Plotting View
        self.tab_plot = QWidget()
        self._setup_plot_tab_layout()

        self.sub_tabs.addTab(self.tab_channels, "Channels")
        self.sub_tabs.addTab(self.tab_plot, "Plot View")
        
        # ==========================================
        # 2. ADD SIGNAL SELECTOR TO THE TAB BAR CORNER
        # ==========================================
        # This replaces the entire top GroupBox and puts it on the tab row
        corner_widget = QWidget()
        h_corner = QHBoxLayout(corner_widget)
        h_corner.setContentsMargins(10, 0, 10, 0) # No vertical padding
        
        lbl_focus = QLabel("<b>Focus Signal:</b>")
        lbl_focus.setStyleSheet(f"color: {self.c_text};")
        
        self.cb_active_signal = QComboBox()
        self.cb_active_signal.setFixedWidth(200)
        self.cb_active_signal.currentTextChanged.connect(self._on_context_changed)
        
        h_corner.addWidget(lbl_focus)
        h_corner.addWidget(self.cb_active_signal)
        
        # Set the widget to the Top Right corner of the tab bar
        self.sub_tabs.setCornerWidget(corner_widget, Qt.Corner.TopRightCorner)
        
        # Add the tab widget as the primary content
        self.main_layout.addWidget(self.sub_tabs)
        
        # Trigger plot update when moving to the plot tab
        self.sub_tabs.currentChanged.connect(self._on_subtab_changed)

    def _setup_plot_tab_layout(self):
        """Builds the internal layout for the plotting sub-tab."""
        layout = QVBoxLayout(self.tab_plot)
        layout.setContentsMargins(0, 5, 0, 0) # Minimize padding

        # --- Controls Area (Now sits at the top of the plot area) ---
        h_controls = QHBoxLayout()
        h_controls.addWidget(QLabel("Window (s):"))
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 500); self.sb_window.setValue(10)
        self.sb_window.valueChanged.connect(self.refresh_plot)
        h_controls.addWidget(self.sb_window)
        
        h_controls.addWidget(QLabel("Ch Visible:"))
        self.sb_n_chans = QSpinBox()
        self.sb_n_chans.setRange(1, 100); self.sb_n_chans.setValue(20)
        self.sb_n_chans.valueChanged.connect(self.refresh_plot)
        h_controls.addWidget(self.sb_n_chans)
        
        h_controls.addWidget(QLabel("Scale (uV):"))
        self.sb_scale = QSpinBox()
        self.sb_scale.setRange(1, 1000000); self.sb_scale.setValue(100)
        self.sb_scale.valueChanged.connect(self.update_plot)
        h_controls.addWidget(self.sb_scale)
        
        self.chk_events = QCheckBox("Events"); self.chk_events.setChecked(True)
        self.chk_events.stateChanged.connect(self.update_plot)
        h_controls.addWidget(self.chk_events)
        
        self.chk_eda = QCheckBox("EDA Results"); self.chk_eda.setChecked(True)
        self.chk_eda.stateChanged.connect(self.update_plot)
        h_controls.addWidget(self.chk_eda)

        h_controls.addStretch()
        self.lbl_coords = QLabel("t=0.000s")
        self.lbl_coords.setStyleSheet(f"font-weight: bold; color: {self.c_text};")
        h_controls.addWidget(self.lbl_coords)
        layout.addLayout(h_controls)

        # --- Plot Area ---
        h_middle = QHBoxLayout()
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.v_scrollbar.valueChanged.connect(self.update_plot)
        h_middle.addWidget(self.v_scrollbar)
        
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.ax = self.figure.add_subplot(111)
        h_middle.addWidget(self.canvas)
        layout.addLayout(h_middle)

        # --- Horizontal Scroll ---
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.valueChanged.connect(self._on_h_scroll)
        layout.addWidget(self.h_scrollbar)

    # --- CORE SYNC LOGIC ---

    def refresh(self):
        """Updates the context switcher and refreshes current sub-tab."""
        if not self.session.workspace: return
        
        # 1. Sync the Context Switcher with Workspace keys
        self.cb_active_signal.blockSignals(True)
        self.cb_active_signal.clear()
        self.cb_active_signal.addItems(list(self.session.workspace.keys()))
        if self.session.active_context:
            self.cb_active_signal.setCurrentText(self.session.active_context)
        self.cb_active_signal.blockSignals(False)

        # 2. Refresh the active sub-tab
        self._sync_current_subtab()

    def _on_context_changed(self, context_name):
        """Updates global focus and refreshes everything."""
        if not context_name: return
        self.session.active_context = context_name
        self._sync_current_subtab()

    def _on_subtab_changed(self, index):
        """Handles internal transitions between Selection and Plot."""
        self._sync_current_subtab()

    def _sync_current_subtab(self):
        """Refreshes whichever internal tab is currently visible."""
        idx = self.sub_tabs.currentIndex()
        if idx == 0: # Channels
            self.tab_channels.refresh()
        else: # Plot
            self.refresh_plot()

    def refresh_plot(self):
        """Recalculates ranges and draws the plot."""
        if not self.session.current_data: return
        self._update_scroll_ranges()
        self.update_plot()

    # --- PLOTTING ENGINE (Improved for Raw vs Epochs) ---

    def update_plot(self):
        if self._updating or not self.session.current_data: return
        self.ax.clear()
        self._setup_axes_style()
        
        # Draw layers
        offsets, picks, times = self._draw_traces()
        if self.chk_events.isChecked(): self._draw_events(times, offsets)
        if self.chk_eda.isChecked(): self._draw_eda_results(offsets)
        
        self.canvas.draw()

    def _setup_axes_style(self):
        self.ax.set_facecolor(self.c_ax)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['left'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(colors=self.c_text)
        self.ax.set_xlabel("Frequency (Hz)", color=self.c_text)
        self.ax.grid(True, linestyle='--', alpha=0.2, color=self.c_spine)

    def _draw_traces(self):
        data_obj = self.session.current_data
        sfreq, all_ch, duration, total_samp = self._get_metadata()
        
        # Resolve which channels to show
        master_list = self.session.selected_channels or all_ch
        start_ch = self.v_scrollbar.value()
        visible_picks = master_list[start_ch : start_ch + self.sb_n_chans.value()]
        
        # Determine sampling range
        start_samp = int(self.current_start_time * sfreq)
        stop_samp = min(int((self.current_start_time + self.sb_window.value()) * sfreq), total_samp)

        try:
            # Check data type to avoid MNE crashes
            if isinstance(data_obj, mne.io.BaseRaw):
                segment, times = data_obj.get_data(picks=visible_picks, start=start_samp, stop=stop_samp, return_times=True)
            else: # Epochs logic
                n_ep_samps = len(data_obj.times)
                idx = start_samp // n_ep_samps
                if idx >= len(data_obj): idx = len(data_obj) - 1
                segment = data_obj.get_data(picks=visible_picks, item=slice(idx, idx+1))
                segment = np.concatenate(segment, axis=1)
                times = np.linspace(idx*(n_ep_samps/sfreq), (idx+1)*(n_ep_samps/sfreq), segment.shape[1])
            
            # Plotting logic
            scale = self.sb_scale.value() * 1e-6
            offsets = np.arange(len(visible_picks))[::-1] * scale
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            for i, ch in enumerate(visible_picks):
                trace = segment[i] - np.mean(segment[i])
                color = colors[i % len(colors)]
                self.ax.plot(times, trace + offsets[i], color=color, lw=0.8)
                self.ax.text(times[0], offsets[i], f" {ch}", color=color, fontweight='bold', va='center')
            
            self.ax.set_xlim(times[0], times[-1])
            if len(offsets) > 0: self.ax.set_ylim(offsets[-1]-scale, offsets[0]+scale*1.5)
            return offsets, visible_picks, times
        except: return [], [], []

    def _draw_events(self, times, offsets):
        """Draws vertical event lines and labels."""
        if not hasattr(self.session, 'obtain_events'): return
        events, event_id = self.session.obtain_events()
        if len(events) == 0: return
        
        sfreq = self.session.current_data.info['sfreq']
        id_to_label = {v: k for k, v in event_id.items()}

        mask = (events[:, 0] / sfreq >= times[0]) & (events[:, 0] / sfreq <= times[-1])
        for ev in events[mask]:
            t_ev = ev[0] / sfreq
            color = self.event_colors[ev[2] % len(self.event_colors)]
            self.ax.axvline(t_ev, color=color, ls='--', lw=1.5, alpha=0.8)
            y_pos = offsets[0] if len(offsets) > 0 else 0
            self.ax.text(t_ev, y_pos, id_to_label.get(ev[2], str(ev[2])), 
                        color=color, rotation=90, va='bottom', fontweight='bold')

    def _draw_eda_results(self, offsets):
        """Draws red spans and diagnostic tags from EDA results."""
        report = getattr(self.session, 'eda_report', None)
        if not report or 'artifacts' not in report: return
        
        arts = report['artifacts']
        bad_indices = arts.get('bad_indices', [])
        bad_tags = arts.get('bad_tags', [])
        tmin, tmax = arts.get('tmin', -1), arts.get('tmax', 3)
        
        events = mne.find_events(self.session.current_data, stim_channel='stim', verbose=False)
        sfreq = self.session.current_data.info['sfreq']
        _, y_max = self.ax.get_ylim()

        for i, idx in enumerate(bad_indices):
            if idx < len(events):
                onset_sec = events[idx, 0] / sfreq
                self.ax.axvspan(onset_sec + tmin, onset_sec + tmax, color='red', alpha=0.2, zorder=0)
                
                if i < len(bad_tags):
                    tag_str = "\n".join(bad_tags[i])
                    self.ax.text(onset_sec + (tmin+tmax)/2, y_max * 0.9, tag_str, 
                                 color='red', fontsize=7, fontweight='bold', ha='center',
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # --- HELPERS ---

    def _get_metadata(self):
        data = self.session.current_data
        sfreq = data.info['sfreq']
        ch_names = data.ch_names
        total_samples = data.n_times
        duration = data.times[-1]
        return sfreq, ch_names, duration, total_samples

    def _update_scroll_ranges(self):
        self._updating = True
        sfreq, all_ch, duration, _ = self._get_metadata()
        
        # Horizontal (Time)
        win = self.sb_window.value()
        self.h_scrollbar.setRange(0, int(max(0, duration - win)))
        self.h_scrollbar.setPageStep(win)
        
        # Vertical (Channels)
        n_total = len(self.session.selected_channels or all_ch)
        self.v_scrollbar.setRange(0, max(0, n_total - self.sb_n_chans.value()))
        self._updating = False

    def _on_h_scroll(self, val):
        self.current_start_time = float(val)
        self.update_plot()

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and event.xdata:
            self.lbl_coords.setText(f"t={event.xdata:.3f}s")