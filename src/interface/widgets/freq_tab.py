import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSpinBox, QPushButton, QComboBox, QSizePolicy, QCheckBox)

class FreqDomainTab(QWidget):
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        
        # Style config
        self.style_kwargs = style_kwargs or {}
        styles = self.style_kwargs
        self.c_bg = styles.get("figure.facecolor", "#2b2d30")
        self.c_ax = styles.get("axes.facecolor", "#1e1f22")
        self.c_text = styles.get("text.color", "#cfd8dc")
        self.c_spine = styles.get("axes.edgecolor", "#4e5154")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- CONTROLS ---
        h_ctrl = QHBoxLayout()
        
        h_ctrl.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["welch", "multitaper"])
        h_ctrl.addWidget(self.cb_method)
        
        h_ctrl.addWidget(QLabel("Fmin:"))
        self.sb_fmin = QSpinBox()
        self.sb_fmin.setRange(0, 100)
        self.sb_fmin.setValue(1)
        h_ctrl.addWidget(self.sb_fmin)
        
        h_ctrl.addWidget(QLabel("Fmax:"))
        self.sb_fmax = QSpinBox()
        self.sb_fmax.setRange(10, 500)
        self.sb_fmax.setValue(60)
        h_ctrl.addWidget(self.sb_fmax)
        
        self.chk_db = QCheckBox("Log Scale (dB)")
        self.chk_db.setChecked(True)
        h_ctrl.addWidget(self.chk_db)
        
        btn_compute = QPushButton("Compute PSD")
        btn_compute.clicked.connect(self.update_plot)
        h_ctrl.addWidget(btn_compute)
        
        h_ctrl.addStretch()
        self.layout.addLayout(h_ctrl)

        # --- PLOT ---
        self.figure = Figure(facecolor=self.c_bg)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas)

    def refresh(self):
        # Optional: Auto-compute on tab switch?
        # self.update_plot()
        pass

    def update_plot(self):
        if not self.session.current_data: return
        
        # 1. Prepare Data & Selection
        data_obj = self.session.current_data
        picks = self.session.selected_channels
        
        if not picks: 
            picks = data_obj.ch_names[:5] # Default to first 5 if all are selected to avoid clutter
            
        valid_picks = [c for c in picks if c in data_obj.ch_names]
        if not valid_picks: return

        # 2. Compute PSD
        fmin = self.sb_fmin.value()
        fmax = self.sb_fmax.value()
        method = self.cb_method.currentText()
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(self.c_ax)
        
        # Styling
        ax.spines['bottom'].set_color(self.c_spine)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.c_spine)
        ax.tick_params(colors=self.c_text)
        ax.set_xlabel("Frequency (Hz)", color=self.c_text)
        ax.set_ylabel("Power Spectral Density", color=self.c_text)
        ax.grid(True, linestyle='--', alpha=0.3, color=self.c_spine)

        try:
            # Create a copy to pick channels before computing (saves RAM)
            # Or pass picks to compute_psd if supported (MNE 1.0+)
            spectrum = data_obj.compute_psd(method=method, fmin=fmin, fmax=fmax, picks=valid_picks)
            
            # Get data: (n_channels, n_freqs) for Raw, or (n_epochs, n_ch, n_freqs) for Epochs
            psd_data, freqs = spectrum.get_data(return_freqs=True)
            
            # If Epochs, average over epochs
            if psd_data.ndim == 3:
                psd_data = psd_data.mean(axis=0)
                
            # Convert to dB
            if self.chk_db.isChecked():
                psd_data = 10 * np.log10(psd_data)
                ax.set_ylabel("Power (dB/Hz)", color=self.c_text)

            # 3. Plot Loop
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            
            for i, ch_name in enumerate(valid_picks):
                color = colors[i % len(colors)]
                ax.plot(freqs, psd_data[i], color=color, label=ch_name, lw=1.5)
            
            # Legend (Styled)
            leg = ax.legend(loc='upper right', facecolor=self.c_ax, edgecolor=self.c_spine)
            for text in leg.get_texts():
                text.set_color(self.c_text)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", color='red', ha='center')

        self.canvas.draw()