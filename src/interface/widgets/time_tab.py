import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollBar, QSpinBox, QCheckBox, QPushButton, 
                             QSizePolicy, QGroupBox, QScrollArea, QFrame)
from PyQt6.QtCore import Qt

class TimeDomainTab(QWidget):
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        
        # --- STYLE PARSING ---
        self.style_kwargs = style_kwargs or {}
        
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_line = self.style_kwargs.get("lines.color", self.c_text)
        
        # Colors for events (cycle through standard matplotlib colors)
        self.event_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']

        plt.rcParams.update(self.style_kwargs)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- TOP CONTROLS ---
        h_controls = QHBoxLayout()
        h_controls.setContentsMargins(0, 0, 0, 0)
        
        # 1. Window (Time)
        h_controls.addWidget(QLabel("Window (s):"))
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 60)
        self.sb_window.setValue(10)
        self.sb_window.valueChanged.connect(self.refresh_scroll_range)
        h_controls.addWidget(self.sb_window)
        
        # 2. Scale (uV)
        h_controls.addWidget(QLabel("Scale (uV):"))
        self.sb_scale = QSpinBox()
        self.sb_scale.setRange(10, 5000)
        self.sb_scale.setSingleStep(10)
        self.sb_scale.setValue(50) 
        self.sb_scale.valueChanged.connect(self.update_plot)
        h_controls.addWidget(self.sb_scale)
        
        # 3. N Channels (Vertical Viewport)
        h_controls.addWidget(QLabel("N Channels:"))
        self.sb_n_chans = QSpinBox()
        self.sb_n_chans.setRange(1, 128)
        self.sb_n_chans.setValue(20) # Default show 20 channels
        self.sb_n_chans.valueChanged.connect(self.refresh_scroll_range)
        h_controls.addWidget(self.sb_n_chans)

        # 4. Events Panel (Scrollable Horizontal Box)
        # We put this inside a GroupBox to organize it
        grp_ev = QGroupBox("Events")
        grp_ev.setStyleSheet(f"color: {self.c_text}; border: 1px solid {self.c_spine}; margin-top: 0px;")
        layout_ev = QHBoxLayout()
        layout_ev.setContentsMargins(2, 2, 2, 2)
        grp_ev.setLayout(layout_ev)
        
        # Scroll area for event checkboxes
        self.scroll_events = QScrollArea()
        self.scroll_events.setWidgetResizable(True)
        self.scroll_events.setFixedHeight(40) # Compact height
        self.scroll_events.setStyleSheet("background: transparent; border: none;")
        self.w_events_container = QWidget()
        self.layout_events_container = QHBoxLayout()
        self.layout_events_container.setContentsMargins(0,0,0,0)
        self.w_events_container.setLayout(self.layout_events_container)
        self.scroll_events.setWidget(self.w_events_container)
        
        layout_ev.addWidget(self.scroll_events)
        h_controls.addWidget(grp_ev)
        
        h_controls.addStretch()
        self.layout.addLayout(h_controls)

        # --- MIDDLE AREA (Plot + V Scroll) ---
        h_middle = QHBoxLayout()
        
        # Canvas
        self.figure = Figure(facecolor=self.c_bg)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.c_ax)
        # Hide standard spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        
        h_middle.addWidget(self.canvas)
        
        # Vertical Scrollbar (Channels)
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.v_scrollbar.valueChanged.connect(self.update_plot)
        h_middle.addWidget(self.v_scrollbar)
        
        self.layout.addLayout(h_middle)

        # --- BOTTOM AREA (Time Scroll) ---
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.setRange(0, 100)
        self.h_scrollbar.valueChanged.connect(self.on_h_scroll)
        self.h_scrollbar.sliderMoved.connect(self.on_h_scroll)
        self.layout.addWidget(self.h_scrollbar)
        
        # State
        self.current_start_time = 0.0
        self.event_checkboxes = {} # {event_id: (QCheckBox, color_hex)}

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def refresh(self):
        """Re-scan data, populate events, reset ranges."""
        self.populate_event_checkboxes()
        self.refresh_scroll_range()
        self.update_plot()

    def populate_event_checkboxes(self):
        """Scans Raw object for events and creates dynamic checkboxes."""
        # Clear old
        while self.layout_events_container.count():
            item = self.layout_events_container.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.event_checkboxes = {}
        
        if not self.session.current_data: return
        data_obj = self.session.current_data
        
        # Only for Raw currently
        if not isinstance(data_obj, mne.io.BaseRaw): return
        
        try:
            # Get unique event IDs present in the file
            events, event_id_map = mne.events_from_annotations(data_obj, verbose=False)
            unique_ids = np.unique(events[:, 2])
            
            # Invert map: ID -> Label
            id_to_label = {v: k for k, v in event_id_map.items()}
            
            for i, ev_id in enumerate(unique_ids):
                label = id_to_label.get(ev_id, str(ev_id))
                color = self.event_colors[i % len(self.event_colors)]
                
                # Create Checkbox
                chk = QCheckBox(label)
                chk.setChecked(True)
                # Color the text to match the event line
                chk.setStyleSheet(f"color: {color}; font-weight: bold;")
                chk.stateChanged.connect(self.update_plot)
                
                self.layout_events_container.addWidget(chk)
                
                # Store reference
                self.event_checkboxes[ev_id] = (chk, color)
                
            self.layout_events_container.addStretch()
            
        except Exception:
            pass

    def on_param_change(self):
        self.refresh_scroll_range()
        self.update_plot()

    def on_h_scroll(self, value):
        self.current_start_time = float(value)
        self.update_plot()

    def refresh_scroll_range(self):
        """Updates limits for both Time and Channel scrollbars."""
        if not self.session.current_data: return
        
        self.h_scrollbar.blockSignals(True)
        self.v_scrollbar.blockSignals(True)
        
        try:
            # 1. TIME LIMITS
            if isinstance(self.session.current_data, mne.io.BaseRaw):
                duration = self.session.current_data.times[-1]
            else:
                duration = 10 
            
            win_size = self.sb_window.value()
            max_h = int(max(0, duration - win_size))
            
            # Clamp value if window grew
            if self.h_scrollbar.value() > max_h:
                self.h_scrollbar.setValue(max_h)
                self.current_start_time = float(max_h)
                
            self.h_scrollbar.setRange(0, max_h)
            self.h_scrollbar.setPageStep(int(win_size))
            
            # 2. CHANNEL LIMITS
            # Get valid list of channels (All or Selection)
            all_picks = self.session.selected_channels
            if not all_picks:
                all_picks = self.session.current_data.ch_names
            
            n_total = len(all_picks)
            n_visible = self.sb_n_chans.value()
            
            max_v = max(0, n_total - n_visible)
            
            if self.v_scrollbar.value() > max_v:
                self.v_scrollbar.setValue(max_v)
                
            self.v_scrollbar.setRange(0, max_v)
            self.v_scrollbar.setPageStep(1) # Scroll 1 channel at a time
            
        finally:
            self.h_scrollbar.blockSignals(False)
            self.v_scrollbar.blockSignals(False)
            
        self.update_plot()

    def update_plot(self):
        """Main Drawing Logic with 2D Slicing."""
        # --- GUARDS ---
        if not self.session.current_data: return
        
        # 1. Determine Channel Set (Full List)
        all_picks = self.session.selected_channels
        if not all_picks: 
            all_picks = self.session.current_data.ch_names
            
        if not all_picks:
            self.ax.clear()
            self.canvas.draw()
            return

        # 2. Slice Channels (Vertical Viewport)
        start_ch_idx = self.v_scrollbar.value()
        n_vis = self.sb_n_chans.value()
        # Safe slicing
        end_ch_idx = min(start_ch_idx + n_vis, len(all_picks))
        visible_picks = all_picks[start_ch_idx : end_ch_idx]
        
        if not visible_picks: return

        # 3. Slice Time (Horizontal Viewport)
        start_t = self.current_start_time
        stop_t = start_t + self.sb_window.value()
        
        data_obj = self.session.current_data
        
        try:
            if isinstance(data_obj, mne.io.BaseRaw):
                sfreq = data_obj.info['sfreq']
                start_samp = int(start_t * sfreq)
                stop_samp = int(stop_t * sfreq)
                stop_samp = min(stop_samp, data_obj.n_times)
                
                # Fetch Data
                data, times = data_obj.get_data(
                    picks=visible_picks,
                    start=start_samp,
                    stop=stop_samp,
                    return_times=True
                )
            else:
                return
        except:
            return

        if data.shape[1] == 0: return

        # --- DRAWING ---
        self.ax.clear()
        
        # Re-apply styles after clear
        self.ax.set_facecolor(self.c_ax)
        self.figure.patch.set_facecolor(self.c_bg)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        self.ax.set_xlabel("Time (s)", color=self.c_text)

        # Plot Signals
        n_channels = len(visible_picks)
        scale_uv = self.sb_scale.value()
        scale_v = scale_uv * 1e-6
        
        offsets = np.arange(n_channels) * scale_v
        offsets = offsets[::-1] # Top channel first
        
        for i in range(n_channels):
            # Center trace
            trace = data[i] - np.mean(data[i])
            self.ax.plot(times, trace + offsets[i], color=self.c_line, lw=0.8)
            
            # Label
            self.ax.text(start_t, offsets[i], f" {visible_picks[i]}", 
                         color=self.c_text, fontsize=9, va='center', fontweight='bold')

        # Plot Events
        if isinstance(data_obj, mne.io.BaseRaw):
            # We fetch all events, then filter by time AND checkbox state
            events, _ = mne.events_from_annotations(data_obj, verbose=False)
            
            # Time filter
            mask_time = (events[:, 0] >= start_samp) & (events[:, 0] <= stop_samp)
            local_events = events[mask_time]
            
            for ev in local_events:
                ev_id = ev[2]
                
                # Check if this event ID exists in our UI map and is Checked
                if ev_id in self.event_checkboxes:
                    chk, color = self.event_checkboxes[ev_id]
                    
                    if chk.isChecked():
                        ev_t = ev[0] / sfreq
                        
                        # Draw line
                        self.ax.axvline(x=ev_t, color=color, linestyle='--', alpha=0.8, lw=1.5)
                        
                        # Draw Label (at top of plot)
                        top_y = offsets[0] + scale_v if len(offsets)>0 else 0
                        label_text = chk.text() # Get text from checkbox
                        self.ax.text(ev_t, top_y, label_text, color=color, 
                                     rotation=90, va='bottom', fontsize=8)

        # Limits
        self.ax.set_xlim(times[0], times[-1])
        if len(offsets) > 0:
            self.ax.set_ylim(offsets[-1] - scale_v, offsets[0] + scale_v)
            
        self.figure.tight_layout()
        self.canvas.draw()