import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSpinBox, QPushButton, QComboBox, QSizePolicy, 
                             QCheckBox, QApplication, QTabWidget)
from PyQt6.QtCore import Qt

# Import your existing ChannelTab implementation
from src.interface.widgets.channel_tab import ChannelTab

class FreqDomainTab(QWidget):
    """
    Advanced Frequency Domain Analysis Container.
    Layout:
    - Corner Widget: Signal Switcher (Focus Signal).
    - Sub-Tab 1: Channel Selection (Reusing ChannelTab).
    - Sub-Tab 2: PSD Visualization (Computation & Plot).
    """
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # --- STYLE PARSING ---
        self.c_bg = self.style_kwargs.get("figure.facecolor", "#2b2d30")
        self.c_ax = self.style_kwargs.get("axes.facecolor", "#1e1f22")
        self.c_text = self.style_kwargs.get("text.color", "#cfd8dc")
        self.c_spine = self.style_kwargs.get("axes.edgecolor", "#4e5154")

        self._init_ui()

    def _init_ui(self):
        """Initializes the layout using QTabWidget with a Corner Widget switcher."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. INTERNAL SUB-TABS
        self.sub_tabs = QTabWidget()
        
        # Sub-Tab A: Channel Selector
        self.tab_channels = ChannelTab(self.session, self.style_kwargs)
        
        # Sub-Tab B: Computation & Plotting
        self.tab_plot = QWidget()
        self._setup_plot_tab_layout()

        self.sub_tabs.addTab(self.tab_channels, "Channels")
        self.sub_tabs.addTab(self.tab_plot, "PSD Plot")
        
        # 2. CORNER WIDGET (Context Switcher)
        corner_container = QWidget()
        h_corner = QHBoxLayout(corner_container)
        h_corner.setContentsMargins(10, 0, 10, 0)
        
        lbl_focus = QLabel("<b>Focus Signal:</b>")
        lbl_focus.setStyleSheet(f"color: {self.c_text};")
        
        self.cb_active_signal = QComboBox()
        self.cb_active_signal.setFixedWidth(200)
        self.cb_active_signal.currentTextChanged.connect(self._on_context_changed)
        
        h_corner.addWidget(lbl_focus)
        h_corner.addWidget(self.cb_active_signal)
        
        # Set to Top Right of the tab bar
        self.sub_tabs.setCornerWidget(corner_container, Qt.Corner.TopRightCorner)
        
        self.main_layout.addWidget(self.sub_tabs)

    def _setup_plot_tab_layout(self):
        """Layout for the actual Power Spectral Density calculation."""
        layout = QVBoxLayout(self.tab_plot)
        layout.setContentsMargins(0, 5, 0, 0)

        # --- Controls Bar ---
        h_ctrl = QHBoxLayout()
        
        h_ctrl.addWidget(QLabel("Method:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems(["welch", "multitaper"])
        h_ctrl.addWidget(self.cb_method)
        
        h_ctrl.addWidget(QLabel("Fmin (Hz):"))
        self.sb_fmin = QSpinBox()
        self.sb_fmin.setRange(0, 200); self.sb_fmin.setValue(1)
        h_ctrl.addWidget(self.sb_fmin)
        
        h_ctrl.addWidget(QLabel("Fmax (Hz):"))
        self.sb_fmax = QSpinBox()
        self.sb_fmax.setRange(1, 1000); self.sb_fmax.setValue(70)
        h_ctrl.addWidget(self.sb_fmax)
        
        self.chk_db = QCheckBox("Log (dB)")
        self.chk_db.setChecked(True)
        h_ctrl.addWidget(self.chk_db)
        
        self.btn_compute = QPushButton("Compute PSD")
        self.btn_compute.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self.update_plot)
        h_ctrl.addWidget(self.btn_compute)
        
        h_ctrl.addStretch()
        layout.addLayout(h_ctrl)

        # --- Plot Canvas ---
        self.figure = Figure(facecolor=self.c_bg, tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        
        layout.addWidget(self.canvas)

    # --- CORE SYNC LOGIC ---

    def refresh(self):
        """Synchronizes the corner switcher and updates current sub-tab."""
        if not self.session.workspace: return
        
        # 1. Update Context Switcher from workspace
        self.cb_active_signal.blockSignals(True)
        self.cb_active_signal.clear()
        self.cb_active_signal.addItems(list(self.session.workspace.keys()))
        if self.session.active_context:
            self.cb_active_signal.setCurrentText(self.session.active_context)
        self.cb_active_signal.blockSignals(False)

        # 2. Refresh active sub-tab
        if self.sub_tabs.currentIndex() == 0:
            self.tab_channels.refresh()

    def _on_context_changed(self, context_name):
        if not context_name: return
        self.session.active_context = context_name
        
        # Refresh current view based on new focus
        if self.sub_tabs.currentIndex() == 0:
            self.tab_channels.refresh()
        else:
            self.ax.clear()
            self.canvas.draw() # Clear old plot to indicate context change

    # --- COMPUTATION ENGINE ---

    def update_plot(self):
        if not self.session.current_data: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            self.ax.clear()
            self._setup_axes_style()
            
            data_obj = self.session.current_data
            # Filter picks to ensure they match current context names
            picks = self.session.selected_channels or data_obj.ch_names[:5]
            valid_picks = [c for c in picks if c in data_obj.ch_names]
            
            if not valid_picks:
                self.ax.text(0.5, 0.5, "No Channels Selected", color='orange', ha='center')
                return

            # Compute Spectrum
            spectrum = data_obj.compute_psd(
                method=self.cb_method.currentText(),
                fmin=self.sb_fmin.value(),
                fmax=self.sb_fmax.value(),
                picks=valid_picks,
                verbose=False
            )
            
            # Extract data using the 'all' fix to avoid MNE filtering non-EEG channels twice
            psd_data, freqs = spectrum.get_data(picks='all', return_freqs=True)
            if psd_data.ndim == 3: psd_data = psd_data.mean(axis=0)
            
            # Labeling and DB conversion
            ylabel = "PSD (V²/Hz)"
            if self.chk_db.isChecked():
                psd_data = 10 * np.log10(np.maximum(psd_data, 1e-20))
                ylabel = "Power (dB/Hz)"

            self._draw_psd_lines(freqs, psd_data, valid_picks, ylabel)

        except Exception as e:
            self.ax.text(0.5, 0.5, f"PSD Computation Error:\n{str(e)}", color='red', ha='center', transform=self.ax.transAxes)
        finally:
            QApplication.restoreOverrideCursor()
            self.canvas.draw()

    def _setup_axes_style(self):
        self.ax.set_facecolor(self.c_ax)
        for spine in ['bottom', 'left']: self.ax.spines[spine].set_color(self.c_spine)
        for spine in ['top', 'right']: self.ax.spines[spine].set_visible(False)
        self.ax.tick_params(colors=self.c_text)
        self.ax.set_xlabel("Frequency (Hz)", color=self.c_text)
        self.ax.grid(True, linestyle='--', alpha=0.2, color=self.c_spine)

    def _draw_psd_lines(self, freqs, psd_data, labels, ylabel):
        self.ax.set_ylabel(ylabel, color=self.c_text)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for i in range(len(psd_data)):
            color = colors[i % len(colors)]
            self.ax.plot(freqs, psd_data[i], color=color, label=labels[i], lw=1.5, alpha=0.85)
        
        if len(labels) <= 15:
            leg = self.ax.legend(loc='upper right', facecolor=self.c_ax, edgecolor=self.c_spine, fontsize=8)
            for text in leg.get_texts(): text.set_color(self.c_text)