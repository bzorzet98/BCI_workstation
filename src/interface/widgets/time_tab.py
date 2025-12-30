import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollBar, QSpinBox, QCheckBox, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt

class TimeDomainTab(QWidget):
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        
        # --- STYLE PARSING ---
        self.style_kwargs = style_kwargs or {}
        styles = self.style_kwargs
        
        self.c_bg = styles.get("figure.facecolor", "#2b2d30")
        self.c_ax = styles.get("axes.facecolor", "#1e1f22")
        self.c_text = styles.get("text.color", "#cfd8dc")
        self.c_spine = styles.get("axes.edgecolor", "#4e5154")
        
        self.event_colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6', '#3498db', '#e67e22']

        # --- RECURSION GUARD ---
        self._updating = False

        # --- LAYOUT SETUP ---
        self.main_layout = QVBoxLayout(self) 

        # --- 1. TOP CONTROLS ---
        h_controls = QHBoxLayout()
        h_controls.setContentsMargins(0, 0, 0, 0)
        
        h_controls.addWidget(QLabel("Window (s):"))
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 500)
        self.sb_window.setValue(10)
        self.sb_window.valueChanged.connect(self.on_param_change)
        h_controls.addWidget(self.sb_window)
        
        h_controls.addWidget(QLabel("Ch Visible:"))
        self.sb_n_chans = QSpinBox()
        self.sb_n_chans.setRange(1, 300)
        self.sb_n_chans.setValue(5) 
        self.sb_n_chans.valueChanged.connect(self.on_param_change)
        h_controls.addWidget(self.sb_n_chans)
        
        h_controls.addWidget(QLabel("Scale (uV):"))
        self.sb_scale = QSpinBox()
        self.sb_scale.setRange(10, 20000)
        self.sb_scale.setSingleStep(100)
        self.sb_scale.setValue(2000)
        self.sb_scale.valueChanged.connect(self.update_plot)
        h_controls.addWidget(self.sb_scale)
        
        self.chk_events = QCheckBox("Show Events")
        self.chk_events.setChecked(True)
        self.chk_events.stateChanged.connect(self.update_plot)
        h_controls.addWidget(self.chk_events)
        
        # Coordinate Label
        h_controls.addStretch()
        self.lbl_coords = QLabel("t=0.00s")
        self.lbl_coords.setStyleSheet(f"font-weight: bold; color: {self.c_text};")
        h_controls.addWidget(self.lbl_coords)
        
        self.main_layout.addLayout(h_controls)

        # --- 2. MIDDLE AREA (Scrollbar LEFT + Canvas) ---
        h_middle = QHBoxLayout()
        h_middle.setContentsMargins(0,0,0,0)
        
        # A. Vertical Scrollbar (LEFT)
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.v_scrollbar.setRange(0, 0) 
        self.v_scrollbar.valueChanged.connect(self.on_v_scroll) # Changed to custom slot
        h_middle.addWidget(self.v_scrollbar)
        
        # B. Canvas (RIGHT)
        self.figure = Figure(facecolor=self.c_bg)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        h_middle.addWidget(self.canvas)
        
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.c_ax)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        
        self.main_layout.addLayout(h_middle)

        # --- 3. BOTTOM AREA (Time Scrollbar) ---
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.setRange(0, 100)
        self.h_scrollbar.valueChanged.connect(self.on_h_scroll)
        self.h_scrollbar.sliderMoved.connect(self.on_h_scroll)
        self.main_layout.addWidget(self.h_scrollbar)
        
        self.current_start_time = 0.0

    # --- EVENT HANDLERS ---

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            time = event.xdata
            if time is not None:
                self.lbl_coords.setText(f"t={time:.3f} s")
        else:
            self.lbl_coords.setText("")

    def refresh(self):
        """Called externally to reset state."""
        self.refresh_scroll_range()
        self.update_plot()

    def on_param_change(self):
        """Called when spinboxes change."""
        self.refresh_scroll_range()
        self.update_plot()

    def on_v_scroll(self, value):
        """Called when vertical scrollbar moves."""
        if self._updating: return
        self.update_plot()

    def on_h_scroll(self, value):
        """Called when horizontal scrollbar moves."""
        if self._updating: return
        self.current_start_time = float(value)
        self.update_plot()

    def refresh_scroll_range(self):
        """Calculates limits for BOTH scrollbars."""
        if not self.session.current_data: return
        
        # Enable Guard
        self._updating = True
        
        try:
            # 1. TIME LIMITS
            data_obj = self.session.current_data
            if isinstance(data_obj, mne.io.BaseRaw):
                duration = data_obj.times[-1]
            else:
                duration = 10 
            
            win_size = self.sb_window.value()
            max_h = int(max(0, duration - win_size))
            
            # Clamp value
            current_h = self.h_scrollbar.value()
            if current_h > max_h:
                self.h_scrollbar.setValue(max_h)
                self.current_start_time = float(max_h)
            
            self.h_scrollbar.setRange(0, max_h)
            self.h_scrollbar.setPageStep(int(win_size))
            
            # 2. CHANNEL LIMITS
            all_picks = self.session.selected_channels
            if not all_picks:
                all_picks = self.session.current_data.ch_names
            
            n_total = len(all_picks)
            n_visible = self.sb_n_chans.value()
            
            max_v = max(0, n_total - n_visible)
            
            current_v = self.v_scrollbar.value()
            if current_v > max_v:
                self.v_scrollbar.setValue(max_v)
                
            self.v_scrollbar.setRange(0, max_v)
            self.v_scrollbar.setPageStep(1)
            
        finally:
            # Disable Guard
            self._updating = False

    def update_plot(self):
        """Main Drawing Logic."""
        if self._updating: return # Prevent recursion
        if not self.session.current_data: return
        
        # 1. Channel Logic
        all_picks = self.session.selected_channels
        data_obj = self.session.current_data
        
        if not all_picks:
            all_picks = data_obj.ch_names
            
        if not all_picks:
            self.ax.clear()
            self.canvas.draw()
            return

        # 2. Viewport Slicing
        start_ch_idx = self.v_scrollbar.value()
        n_vis = self.sb_n_chans.value()
        end_ch_idx = min(start_ch_idx + n_vis, len(all_picks))
        valid_picks = all_picks[start_ch_idx : end_ch_idx]
        
        # Validate existence
        valid_picks = [c for c in valid_picks if c in data_obj.ch_names]
        
        if not valid_picks:
            self.ax.clear()
            self.canvas.draw()
            return

        # 3. Time Slicing
        start_t = self.current_start_time
        stop_t = start_t + self.sb_window.value()
        
        try:
            if isinstance(data_obj, mne.io.BaseRaw):
                sfreq = data_obj.info['sfreq']
                start_samp = int(start_t * sfreq)
                stop_samp = int(stop_t * sfreq)
                stop_samp = min(stop_samp, data_obj.n_times)
                
                segment, times = data_obj.get_data(
                    picks=valid_picks,
                    start=start_samp, 
                    stop=stop_samp, 
                    return_times=True
                )
            else:
                return 
        except Exception:
            return

        if segment.shape[1] == 0: return

        # 4. PLOTTING
        self.ax.clear()
        
        # Reset Styles
        self.ax.set_facecolor(self.c_ax)
        self.figure.patch.set_facecolor(self.c_bg)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        self.ax.set_xlabel("Time (s)", color=self.c_text)

        n_channels = len(valid_picks)
        scale_uv = self.sb_scale.value()
        scale_v = scale_uv * 1e-6
        
        # Offsets
        offsets = np.arange(n_channels) * scale_v
        offsets = offsets[::-1]
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        for i in range(n_channels):
            trace = segment[i] - np.mean(segment[i])
            line_color = colors[i % len(colors)]
            
            self.ax.plot(times, trace + offsets[i], color=line_color, lw=0.8)
            
            # Label
            self.ax.text(start_t, offsets[i], f" {valid_picks[i]}", 
                         color=line_color, 
                         fontsize=9, va='center', fontweight='bold')

        # 5. Events
        if self.chk_events.isChecked() and isinstance(data_obj, mne.io.BaseRaw):
            events, event_id_dict = self.session.obtain_events()
            
            if len(events) > 0:
                mask = (events[:, 0] >= start_samp) & (events[:, 0] <= stop_samp)
                visible_events = events[mask]
                id_to_label = {v: k for k, v in event_id_dict.items()}
                
                for ev in visible_events:
                    sample = ev[0]
                    ev_id = ev[2]
                    t_ev = sample / sfreq
                    label = id_to_label.get(ev_id, str(ev_id))
                    color = self.event_colors[ev_id % len(self.event_colors)]
                    
                    self.ax.axvline(x=t_ev, color=color, linestyle='--', alpha=0.9, lw=1.5)
                    
                    top_y = offsets[0] + scale_v if len(offsets) > 0 else 0
                    self.ax.text(t_ev, top_y, label, 
                            color=color, rotation=90, va='bottom', fontsize=10, fontweight='bold')

        # 6. Limits
        self.ax.set_xlim(times[0], times[-1])
        if len(offsets) > 0:
            self.ax.set_ylim(offsets[-1] - scale_v, offsets[0] + scale_v * 1.5) 

        self.figure.tight_layout()
        self.canvas.draw()