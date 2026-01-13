import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSpinBox, QPushButton, QComboBox, QSizePolicy, 
                             QCheckBox, QApplication)
from PyQt6.QtCore import Qt

class FreqDomainTab(QWidget):
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

        self._init_ui()

    def _init_ui(self):
        """Builds the control bar and the PSD canvas."""
        self.main_layout = QVBoxLayout(self)

        # 1. TOP CONTROLS
        h_ctrl = QHBoxLayout()
        
        h_ctrl.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["welch", "multitaper"])
        h_ctrl.addWidget(self.cb_method)
        
        h_ctrl.addWidget(QLabel("Fmin (Hz):"))
        self.sb_fmin = QSpinBox()
        self.sb_fmin.setRange(0, 200)
        self.sb_fmin.setValue(1)
        h_ctrl.addWidget(self.sb_fmin)
        
        h_ctrl.addWidget(QLabel("Fmax (Hz):"))
        self.sb_fmax = QSpinBox()
        self.sb_fmax.setRange(1, 1000)
        self.sb_fmax.setValue(70)
        h_ctrl.addWidget(self.sb_fmax)
        
        self.chk_db = QCheckBox("Log Scale (dB)")
        self.chk_db.setChecked(True)
        h_ctrl.addWidget(self.chk_db)
        
        self.btn_compute = QPushButton("Compute PSD")
        self.btn_compute.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self.update_plot)
        h_ctrl.addWidget(self.btn_compute)
        
        h_ctrl.addStretch()
        self.main_layout.addLayout(h_ctrl)

        # 2. PSD PLOT
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        
        self.main_layout.addWidget(self.canvas)

    # --- PUBLIC API ---

    def refresh(self):
        """Called by the main window when data or channels change."""
        # We don't auto-compute PSD as it can be heavy; we let the user click the button.
        pass

    def update_plot(self):
        """Main controller for PSD computation and visualization."""
        if not self.session.current_data: return
        
        # 1. Feedback: Set Wait Cursor (Multitaper can be slow)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            self.ax.clear()
            self._setup_axes_style()
            
            # 2. Channel Selection
            data_obj = self.session.current_data
            picks = self.session.selected_channels or data_obj.ch_names[:5]
            valid_picks = [c for c in picks if c in data_obj.ch_names]
            
            if not valid_picks:
                self.ax.text(0.5, 0.5, "No valid channels selected", color='orange', ha='center')
                return

            # 3. Compute PSD
            # MNE 1.2+ uses Spectrum objects
            spectrum = data_obj.compute_psd(
                method=self.cb_method.currentText(),
                fmin=self.sb_fmin.value(),
                fmax=self.sb_fmax.value(),
                picks=valid_picks,
                verbose=False
            )
            
            # 4. Extract and Transform Data
            # Note: picks='all' is used here because the spectrum object 
            # ALREADY contains only our valid_picks. This prevents MNE 
            # from filtering out EMG/EOG in the get_data step.
            psd_data, freqs = spectrum.get_data(picks='all', return_freqs=True)
            
            # Average across epochs if necessary
            if psd_data.ndim == 3:
                psd_data = psd_data.mean(axis=0)
            
            # Log Scaling
            ylabel = "PSD (V²/Hz)"
            if self.chk_db.isChecked():
                psd_data = 10 * np.log10(np.maximum(psd_data, 1e-20))
                ylabel = "Power (dB/Hz)"

            # 5. Drawing
            self._draw_psd_lines(freqs, psd_data, valid_picks, ylabel)

        except Exception as e:
            self.ax.text(0.5, 0.5, f"PSD Error:\n{str(e)}", color='red', ha='center', va='center', transform=self.ax.transAxes)
        
        finally:
            QApplication.restoreOverrideCursor()
            self.canvas.draw()

    # --- PRIVATE HELPERS ---

    def _setup_axes_style(self):
        self.ax.set_facecolor(self.c_ax)
        self.ax.spines['bottom'].set_color(self.c_spine)
        self.ax.spines['left'].set_color(self.c_spine)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.tick_params(colors=self.c_text)
        self.ax.set_xlabel("Frequency (Hz)", color=self.c_text)
        self.ax.grid(True, linestyle='--', alpha=0.2, color=self.c_spine)

    def _draw_psd_lines(self, freqs, psd_data, labels, ylabel):
        self.ax.set_ylabel(ylabel, color=self.c_text)
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(len(psd_data)):
            color = colors[i % len(colors)]
            name = labels[i] if i < len(labels) else f"Ch{i}"
            self.ax.plot(freqs, psd_data[i], color=color, label=name, lw=1.5, alpha=0.85)
        
        if len(labels) <= 15:
            leg = self.ax.legend(loc='upper right', facecolor=self.c_ax, edgecolor=self.c_spine, fontsize=8)
            for text in leg.get_texts():
                text.set_color(self.c_text)