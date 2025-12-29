from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox
from src.interface.plotter import Plotter

class VizTab(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Group: Time Domain
        grp_time = QGroupBox("Time Domain")
        v_time = QVBoxLayout()
        btn_interactive = QPushButton("Interactive Plot")
        btn_interactive.clicked.connect(self.plot_interactive)
        v_time.addWidget(btn_interactive)
        grp_time.setLayout(v_time)
        self.layout.addWidget(grp_time)
        
        # Group: Frequency Domain
        grp_freq = QGroupBox("Frequency Domain")
        v_freq = QVBoxLayout()
        btn_psd = QPushButton("Plot PSD")
        btn_psd.clicked.connect(self.plot_psd)
        v_freq.addWidget(btn_psd)
        grp_freq.setLayout(v_freq)
        self.layout.addWidget(grp_freq)
        
        # Group: Sensors
        grp_spatial = QGroupBox("Spatial")
        v_spatial = QVBoxLayout()
        btn_sensors = QPushButton("Plot Sensors")
        btn_sensors.clicked.connect(self.plot_sensors)
        v_spatial.addWidget(btn_sensors)
        grp_spatial.setLayout(v_spatial)
        self.layout.addWidget(grp_spatial)
        
        self.layout.addStretch()

    def get_current_picks(self):
        """Helper to get valid picks from session."""
        if not self.session.current_data:
            return None
            
        selected = self.session.selected_channels
        all_chs = self.session.current_data.ch_names
        
        if selected:
            # Validate: Only pass channels that actually exist
            valid_picks = [ch for ch in selected if ch in all_chs]
            if not valid_picks:
                print("Warning: Selection invalid (channels likely dropped). Plotting all.")
                return None # None means "All" in MNE
            return valid_picks
        
        return None # None means "All" in MNE

    def plot_interactive(self):
        picks = self.get_current_picks()
        Plotter.plot_interactive(self.session.current_data, picks=picks)

    def plot_psd(self):
        picks = self.get_current_picks()
        Plotter.plot_psd(self.session.current_data, picks=picks)

    def plot_sensors(self):
        # Sensors plot doesn't usually use picks for the montage view, 
        # but we pass the object directly.
        Plotter.plot_sensors(self.session.current_data)