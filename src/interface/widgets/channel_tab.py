import re
import mne
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QScrollArea, QCheckBox, QGridLayout, QLabel, 
                             QFrame, QGroupBox)
from PyQt6.QtCore import Qt

class ChannelTab(QWidget):
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        
        # Internal State
        self.checkboxes = {} # Map: ch_name -> QCheckBox
        
        self._init_ui()

    def _init_ui(self):
        """Initializes the main structural layout."""
        self.main_layout = QVBoxLayout(self)

        # 1. Global Controls (Top)
        self.main_layout.addWidget(self._create_global_controls())
        
        # 2. Row Selection Buttons (Middle)
        self.grp_rows = QGroupBox("Anatomical Row Selection")
        self.row_layout = QHBoxLayout(self.grp_rows)
        self.main_layout.addWidget(self.grp_rows)

        # 3. Channel Grid (Bottom - Scrollable)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6;")
        
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.scroll.setWidget(self.grid_container)
        
        self.main_layout.addWidget(self.scroll)

    def _create_global_controls(self):
        """Creates the master toggle buttons."""
        grp = QGroupBox("Master Selection")
        layout = QHBoxLayout(grp)

        btns = {
            "All": self._select_all,
            "None": self._select_none,
            "EEG Only": self._select_eeg_only,
            "Left (Odd)": lambda: self._toggle_by_condition(self._is_left),
            "Right (Even)": lambda: self._toggle_by_condition(self._is_right),
            "Midline (z)": lambda: self._toggle_by_condition(lambda ch: 'z' in ch.lower())
        }

        for text, func in btns.items():
            btn = QPushButton(text)
            btn.clicked.connect(func)
            layout.addWidget(btn)
        
        return grp

    # --- PUBLIC API ---

    def refresh(self):
        """Rebuilds the anatomical grid based on current session data."""
        self._clear_ui()
        if not self.session.current_data: return

        ch_names = self.session.current_data.ch_names
        
        # 1. Sort channels into anatomical rows
        rows_order = ['Fp', 'AF', 'F', 'FT/FC', 'T/C', 'TP/CP', 'P', 'PO', 'O', 'I', 'EOG', 'EMG', 'Other']
        categorized = {r: [] for r in rows_order}
        
        for ch in ch_names:
            row = self._parse_anatomical_row(ch)
            categorized.get(row, categorized['Other']).append(ch)

        # 2. Build UI Rows
        current_grid_row = 0
        for row_label, channels in categorized.items():
            if not channels: continue
            
            # Add Shortcut Button to the top panel
            btn = QPushButton(row_label)
            btn.clicked.connect(lambda _, chs=channels: self._toggle_list(chs))
            self.row_layout.addWidget(btn)
            
            # Add Row Label to Grid
            lbl = QLabel(f"<b>{row_label}</b>")
            lbl.setStyleSheet("color: #495057; border-bottom: 1px solid #dee2e6;")
            self.grid_layout.addWidget(lbl, current_grid_row, 0)
            
            # Add Checkboxes
            col = 1
            for ch in channels:
                chk = QCheckBox(ch)
                chk.setChecked(True)
                chk.stateChanged.connect(self._update_session)
                chk.setStyleSheet("color: #212529;")
                
                self.grid_layout.addWidget(chk, current_grid_row, col)
                self.checkboxes[ch] = chk
                
                col += 1
                if col > 8: # Wrap row if more than 8 channels in anatomical group
                    current_grid_row += 1
                    col = 1
            
            current_grid_row += 1
            
        self._update_session()

    # --- LOGIC & HELPERS ---

    def _toggle_list(self, ch_names):
        """Smart Toggle: If all are checked, uncheck all. Otherwise, check all."""
        all_checked = all(self.checkboxes[ch].isChecked() for ch in ch_names if ch in self.checkboxes)
        target = not all_checked
        
        self.blockSignals(True)
        for ch in ch_names:
            if ch in self.checkboxes: self.checkboxes[ch].setChecked(target)
        self.blockSignals(False)
        self._update_session()

    def _toggle_by_condition(self, condition_func):
        """Toggles a subset of channels based on a boolean function (e.g. is_left)."""
        targets = [ch for ch in self.checkboxes.keys() if condition_func(ch)]
        self._toggle_list(targets)

    def _update_session(self):
        """Synchronizes selected channels with the global session state."""
        selected = [ch for ch, box in self.checkboxes.items() if box.isChecked()]
        self.session.selected_channels = selected

    def _parse_anatomical_row(self, ch_name):
        """Determines the row category based on standard 10-20 naming."""
        ch = ch_name.upper()
        if 'EOG' in ch: return 'EOG'
        if 'EMG' in ch: return 'EMG'
        
        # Strip numbers and 'Z' to find base row
        base = ''.join([c for c in ch if c.isalpha() and c != 'Z'])
        
        mapping = {
            'FP': 'Fp', 'AF': 'AF', 'F': 'F',
            'FT': 'FT/FC', 'FC': 'FT/FC',
            'T': 'T/C', 'C': 'T/C',
            'TP': 'TP/CP', 'CP': 'TP/CP',
            'P': 'P', 'PO': 'PO', 'O': 'O', 'I': 'I'
        }
        return mapping.get(base, 'Other')

    def _is_left(self, name):
        nums = re.findall(r'\d+', name)
        return int(nums[0]) % 2 != 0 if nums else False

    def _is_right(self, name):
        nums = re.findall(r'\d+', name)
        return int(nums[0]) % 2 == 0 if nums else False

    # --- MASTER ACTIONS ---

    def _select_all(self):
        for b in self.checkboxes.values(): b.setChecked(True)

    def _select_none(self):
        for b in self.checkboxes.values(): b.setChecked(False)

    def _select_eeg_only(self):
        if not self.session.current_data: return
        self._select_none()
        try:
            eeg_picks = mne.pick_types(self.session.current_data.info, eeg=True)
            eeg_names = [self.session.current_data.ch_names[i] for i in eeg_picks]
            self.blockSignals(True)
            for name in eeg_names:
                if name in self.checkboxes: self.checkboxes[name].setChecked(True)
            self.blockSignals(False)
            self._update_session()
        except: pass

    def _clear_ui(self):
        """Cleans up widgets before a refresh."""
        self.checkboxes = {}
        for layout in [self.grid_layout, self.row_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget(): child.widget().deleteLater()