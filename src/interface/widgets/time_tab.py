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

        # Recursion Guard
        self._updating = False

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- 1. TOP CONTROLS ---
        h_controls = QHBoxLayout()
        h_controls.setContentsMargins(0, 0, 0, 0)
        
        # Window (Time)
        h_controls.addWidget(QLabel("Window (s):"))
        self.sb_window = QSpinBox()
        self.sb_window.setRange(1, 500)
        self.sb_window.setValue(10)
        self.sb_window.valueChanged.connect(self.on_param_change)
        h_controls.addWidget(self.sb_window)
        
        # N Channels
        h_controls.addWidget(QLabel("Ch Visible:"))
        self.sb_n_chans = QSpinBox()
        self.sb_n_chans.setRange(1, 300)
        self.sb_n_chans.setValue(20)
        self.sb_n_chans.valueChanged.connect(self.refresh_scroll_range)
        h_controls.addWidget(self.sb_n_chans)
        
        # Scale Control
        h_controls.addWidget(QLabel("Scale:"))
        self.sb_scale = QSpinBox()
        # FIX: Huge range to allow manual control
        self.sb_scale.setRange(1, 1000000) 
        self.sb_scale.setSingleStep(50)
        self.sb_scale.setValue(50) # Default 50uV
        self.sb_scale.valueChanged.connect(self.update_plot)
        h_controls.addWidget(self.sb_scale)
        
        # FIX: Manual Toggle for Units
        self.chk_zscore = QCheckBox("Z-Score Mode")
        self.chk_zscore.setToolTip("Check this if data is normalized (Std Units). Uncheck for Raw data (uV).")
        self.chk_zscore.setChecked(False) # Default to Raw/uV
        self.chk_zscore.stateChanged.connect(self.update_plot)
        h_controls.addWidget(self.chk_zscore)
        
        # Events
        self.chk_events = QCheckBox("Events")
        self.chk_events.setChecked(True)
        self.chk_events.stateChanged.connect(self.update_plot)
        h_controls.addWidget(self.chk_events)
        
        self.chk_eda = QCheckBox("EDA Results")
        self.chk_events.setChecked(False)
        self.chk_eda.stateChanged.connect(self.refresh)
        h_controls.addWidget(self.chk_eda) # Añádelo junto al de "Events"
        
        # Coordinate Label
        h_controls.addStretch()
        self.lbl_coords = QLabel("t=0.00s")
        self.lbl_coords.setStyleSheet(f"font-weight: bold; color: {self.c_text};")
        h_controls.addWidget(self.lbl_coords)
        
        self.layout.addLayout(h_controls)

        # --- 2. MIDDLE AREA ---
        h_middle = QHBoxLayout()
        
        # A. Vertical Scrollbar (Channels)
        self.v_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self.v_scrollbar.setRange(0, 0)
        self.v_scrollbar.valueChanged.connect(self.on_v_scroll)
        h_middle.addWidget(self.v_scrollbar)
        
        # B. Canvas
        self.figure = Figure(facecolor=self.c_bg)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Mouse Event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Setup Axes Once
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.c_ax)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        
        h_middle.addWidget(self.canvas)
        self.layout.addLayout(h_middle)

        # --- 3. BOTTOM AREA ---
        self.h_scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self.h_scrollbar.setRange(0, 100)
        self.h_scrollbar.valueChanged.connect(self.on_h_scroll)
        self.h_scrollbar.sliderMoved.connect(self.on_h_scroll)
        self.layout.addWidget(self.h_scrollbar)
        
                
        # State
        self.current_start_time = 0.0

    # --- HANDLERS ---

    def on_mouse_move(self, event):
        if event.inaxes == self.ax:
            time = event.xdata
            if time is not None:
                self.lbl_coords.setText(f"t={time:.3f} s")
        else:
            self.lbl_coords.setText("")

    def refresh(self):
        self.refresh_scroll_range()
        self.update_plot()

    def on_param_change(self):
        self.refresh_scroll_range()
        self.update_plot()

    def on_v_scroll(self, value):
        if self._updating: return
        self.update_plot()

    def on_h_scroll(self, value):
        if self._updating: return
        self.current_start_time = float(value)
        self.update_plot()

    def get_metadata_from_current(self):
        data = self.session.current_data
        sfreq = 1.0
        ch_names = []
        duration = 10.0
        n_times = 0
        
        if isinstance(data, mne.io.BaseRaw):
            sfreq = data.info['sfreq']
            ch_names = data.ch_names
            n_times = data.n_times
            duration = data.times[-1]
        elif isinstance(data, (mne.Epochs, mne.EpochsArray)):
            sfreq = data.info['sfreq']
            ch_names = data.ch_names
            n_epochs = len(data)
            n_samples_per_epoch = len(data.times)
            n_times = n_epochs * n_samples_per_epoch
            duration = n_times / sfreq
        elif isinstance(data, dict):
            arr = data.get('data')
            if arr is not None:
                if arr.ndim == 3:
                    n_times = arr.shape[0] * arr.shape[2]
                else:
                    n_times = arr.shape[1]
            sfreq = data.get('sfreq', 128.0)
            duration = n_times / sfreq
            ch_names = data.get('ch_names', [f"Ch{i}" for i in range(arr.shape[-2])]) if arr is not None else []
            
        return sfreq, ch_names, duration, n_times

    def refresh_scroll_range(self):
        if not self.session.current_data: return
        
        self.h_scrollbar.blockSignals(True)
        self.v_scrollbar.blockSignals(True)
        self._updating = True
        
        try:
            sfreq, ch_names, duration, n_times = self.get_metadata_from_current()

            # Time
            win_size = self.sb_window.value()
            max_h = int(max(0, duration - win_size))
            
            if self.h_scrollbar.value() > max_h:
                self.h_scrollbar.setValue(max_h)
                self.current_start_time = float(max_h)
            
            self.h_scrollbar.setRange(0, max_h)
            self.h_scrollbar.setPageStep(int(win_size))
            
            # Channel
            all_picks = self.session.selected_channels
            if not all_picks: all_picks = ch_names
            
            n_total = len(all_picks)
            n_visible = self.sb_n_chans.value()
            max_v = max(0, n_total - n_visible)
            
            if self.v_scrollbar.value() > max_v:
                self.v_scrollbar.setValue(max_v)
                
            self.v_scrollbar.setRange(0, max_v)
            self.v_scrollbar.setPageStep(1)
            
        finally:
            self.h_scrollbar.blockSignals(False)
            self.v_scrollbar.blockSignals(False)
            self._updating = False
        
        self.update_plot()

    def update_plot(self):
        if not self.session.current_data: return
        
        sfreq, all_ch_names, duration, total_samples = self.get_metadata_from_current()
        
        # 1. Master List
        if self.session.selected_channels is not None:
            master_list = [ch for ch in self.session.selected_channels if ch in all_ch_names]
        else:
            master_list = all_ch_names
            
        if not master_list:
            self.ax.clear()
            self.canvas.draw()
            return

        events = mne.find_events(self.session.current_data, stim_channel='stim', verbose=False)
        
        # 2. Slice Channels
        start_idx = self.v_scrollbar.value()
        n_vis = self.sb_n_chans.value()
        end_idx = min(start_idx + n_vis, len(master_list))
        visible_picks = master_list[start_idx : end_idx]
        
        if not visible_picks: return

        # 3. Slice Time
        start_t = self.current_start_time
        stop_t = start_t + self.sb_window.value()
        start_samp = int(start_t * sfreq)
        stop_samp = int(stop_t * sfreq)
        stop_samp = min(stop_samp, total_samples)
        
        data_obj = self.session.current_data
        segment = None
        times = None
        
        try:
            # RAW
            if isinstance(data_obj, mne.io.BaseRaw):
                segment, times = data_obj.get_data(
                    picks=visible_picks,
                    start=start_samp, 
                    stop=stop_samp, 
                    return_times=True
                )
            # EPOCHS
            elif isinstance(data_obj, (mne.Epochs, mne.EpochsArray)):
                n_samples_per_epoch = len(data_obj.times)
                start_epoch_idx = start_samp // n_samples_per_epoch
                end_epoch_idx = stop_samp // n_samples_per_epoch
                
                epoch_data = data_obj.get_data(picks=visible_picks, item=slice(start_epoch_idx, end_epoch_idx + 1))
                segment = np.concatenate(epoch_data, axis=1) 
                
                relative_start = start_samp % n_samples_per_epoch
                req_len = stop_samp - start_samp
                
                if segment.shape[1] > relative_start:
                    segment = segment[:, relative_start : relative_start + req_len]
                else:
                    segment = np.empty((len(visible_picks), 0))
                
                times = np.arange(start_samp, start_samp + segment.shape[1]) / sfreq
            
            # DICT
            elif isinstance(data_obj, dict) and 'data' in data_obj:
                arr = data_obj['data']
                ch_indices = [all_ch_names.index(ch) for ch in visible_picks if ch in all_ch_names]
                
                if arr.ndim == 2:
                    segment = arr[ch_indices, start_samp:stop_samp]
                elif arr.ndim == 3:
                    flat_arr = np.concatenate(arr, axis=1)
                    segment = flat_arr[ch_indices, start_samp:stop_samp]
                
                times = np.arange(start_samp, start_samp + segment.shape[1]) / sfreq

        except Exception as e:
            print(f"Data fetch error: {e}")
            return

        if segment is None or segment.shape[1] == 0: return

        # 4. PLOTTING
        self.ax.clear()
        
        # Styles
        self.ax.set_facecolor(self.c_ax)
        self.figure.patch.set_facecolor(self.c_bg)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(axis='x', colors=self.c_text)
        self.ax.set_yticks([]) 
        self.ax.set_xlabel("Time (s)", color=self.c_text)

        n_channels = len(visible_picks)
        val = self.sb_scale.value()
        
        # --- FIX: MANUAL SCALE LOGIC ---
        if self.chk_zscore.isChecked():
            # Normalized Mode: Scale = 1 unit (approx 1 STD)
            # Example: Value 3 means distance is 3.0 sigmas
            # We divide by a smaller factor because ranges are -3 to +3
            scale_v = val 
        else:
            # Raw Mode (uV): Scale = 1e-6 V
            # Example: Value 50 means distance is 50uV
            scale_v = val * 1e-6
        
        # Offsets
        offsets = np.arange(n_channels) * scale_v
        offsets = offsets[::-1]
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        for i in range(n_channels):
            if i >= segment.shape[0]: break
            
            trace = segment[i] - np.mean(segment[i])
            line_color = colors[i % len(colors)]
            
            self.ax.plot(times, trace + offsets[i], color=line_color, lw=0.8)
            
            # Colored Labels
            self.ax.text(start_t, offsets[i], f" {visible_picks[i]}", 
                         color=line_color, fontsize=9, va='center', fontweight='bold')

        # 5. Events
        if self.chk_events.isChecked() and hasattr(self.session, 'obtain_events'):
            # Restrict event plotting to Raw or compatible types
            if isinstance(data_obj, mne.io.BaseRaw):
                events, event_id_dict = self.session.obtain_events()
                
                if len(events) > 0:
                    mask = (events[:, 0] >= start_samp) & (events[:, 0] <= stop_samp)
                    visible_events = events[mask]
                    id_to_label = {v: k for k, v in event_id_dict.items()}
                    
                    for ev in visible_events:
                        t_ev = ev[0] / sfreq
                        label = id_to_label.get(ev[2], str(ev[2]))
                        color = self.event_colors[ev[2] % len(self.event_colors)]
                        
                        self.ax.axvline(x=t_ev, color=color, linestyle='--', alpha=0.9, lw=1.5)
                        
                        top_y = offsets[0] + scale_v if len(offsets) > 0 else 0
                        self.ax.text(t_ev, top_y, label, 
                                color=color, rotation=90, va='bottom', fontsize=10, fontweight='bold')

        # 6. Limits
        self.ax.set_xlim(times[0], times[-1])
        if len(offsets) > 0:
            self.ax.set_ylim(offsets[-1] - scale_v, offsets[0] + scale_v * 1.5) 

        
        # 7. Resaltar trials identificados por EDA
        if self.chk_eda.isChecked():
            # 1. Recuperar el reporte de la sesión
            report = getattr(self.session, 'eda_report', None)
            
            if report and 'artifacts' in report:
                arts = report['artifacts']
                bad_trials = arts.get('bad_trials', [])
                bad_tags = arts.get('tags', [])
                tmin = arts.get('tmin', -1.0)
                tmax = arts.get('tmax', 3.0)
                
                # 2. Encontrar los eventos para saber su posición temporal exacta
                try:
                    # Buscamos eventos en el canal 'stim' (o el que use tu dataset)
                    events = mne.find_events(self.session.current_data, stim_channel='stim', verbose=False)
                    sfreq = self.session.current_data.info['sfreq']
                    
                    for idx in bad_trials:
                        if idx < len(events):
                            # Tiempo del evento en segundos
                            onset_sec = events[idx, 0] / sfreq
                            
                            # Dibujar el rectángulo rojo (transparente)
                            # zorder=0 lo pone detrás de las señales
                            self.ax.axvspan(onset_sec + tmin, onset_sec + tmax, 
                                            color='red', alpha=0.3, zorder=0)
                        tags = bad_tags[i]
                        # Unimos los tags con saltos de línea si hay varios
                        label_text = "\n".join(tags)
                        
                        y_min, y_max = self.ax.get_ylim()
                        # Ponemos el texto un poco por debajo del límite superior (y_max)
                        self.ax.text(onset_sec + (tmin + tmax)/2, # Centrado en X
                                     y_max * 0.9,                # Al 90% de altura en Y
                                     label_text, 
                                     color='red', 
                                     fontweight='bold',
                                     fontsize=8,
                                     ha='center',                # Alineación horizontal
                                     va='top',                   # Alineación vertical
                                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')) # Fondo para legibilidad

                except Exception as e:
                    print(f"Error al resaltar EDA: {e}")

        self.canvas.draw()
