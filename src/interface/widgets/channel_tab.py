import re
import mne
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QScrollArea, QCheckBox, QGridLayout, QLabel, QFrame, QGroupBox)
from PyQt6.QtCore import Qt

class ChannelTab(QWidget):
    def __init__(self, session):
        super().__init__()
        self.session = session
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # --- 1. Global Controls ---
        grp_global = QGroupBox("Global Selection")
        h_global = QHBoxLayout()
        
        # Global Controls
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_eeg = QPushButton("EEG Only")
        btn_left = QPushButton("Left (Odd)")
        btn_right = QPushButton("Right (Even)")
        btn_center = QPushButton("Center (z)")
        
        # Connections using Toggle Logic
        btn_all.clicked.connect(self.select_all)
        btn_none.clicked.connect(self.select_none)
        
        # EEG Only is special (usually exclusive selection), 
        # but if you want it to toggle, you can adapt it. 
        # For now, "Select EEG Only" usually implies unchecking others, so keep as is:
        btn_eeg.clicked.connect(self.select_eeg) 
        
        # Smart Toggles for Topology
        btn_left.clicked.connect(lambda: self.toggle_by_condition(lambda ch: self.is_left(ch)))
        btn_right.clicked.connect(lambda: self.toggle_by_condition(lambda ch: self.is_right(ch)))
        btn_center.clicked.connect(lambda: self.toggle_by_condition(lambda ch: 'z' in ch.lower()))
        
        h_global.addWidget(btn_all)
        h_global.addWidget(btn_none)
        h_global.addWidget(btn_eeg)
        h_global.addWidget(btn_left)
        h_global.addWidget(btn_right)
        h_global.addWidget(btn_center)
        grp_global.setLayout(h_global)
        self.layout.addWidget(grp_global)
        
        # --- 2. Row Selection Controls ---
        # Buttons to select specific anatomical rows
        grp_rows = QGroupBox("Anatomical Rows")
        self.row_layout = QHBoxLayout() # Will be populated dynamically
        grp_rows.setLayout(self.row_layout)
        self.layout.addWidget(grp_rows)

        # --- 3. Channel Grid (Scrollable) ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        
        self.chk_container = QWidget()
        # We use a Grid where: Col 0 = Label, Col 1..N = Checkboxes
        self.grid_layout = QGridLayout() 
        self.chk_container.setLayout(self.grid_layout)
        
        self.scroll.setWidget(self.chk_container)
        self.layout.addWidget(self.scroll)
        
        self.checkboxes = {} # Map name -> QCheckBox

    # --- LOGIC HELPERS ---
    
    def get_anatomical_row(self, ch_name):
        """
        Parses channel name to find its row.
        """
        ch = ch_name.upper()
        
        # Non-EEG types
        if 'EOG' in ch: return 'EOG'
        if 'EMG' in ch: return 'EMG'
        if 'STIM' in ch: return 'STIM'
        
        # Remove 'z' (midline) and numbers to find the base code
        # Example: 'Fz' -> 'F', 'FCz' -> 'FC', 'F3' -> 'F'
        base = ''.join([c for c in ch if c.isalpha() and c != 'Z'])
        
        # If the channel was JUST 'z' (rare) or stripping resulted in empty, handle it
        if not base and 'Z' in ch: 
            # Fallback logic if needed, but usually Fz becomes F
            pass

        # Map specific bases to rows
        if base == 'FP': return 'Fp'
        if base == 'AF': return 'AF'
        if base == 'F':  return 'F'
        if base in ['FT', 'FC']: return 'FT/FC'
        if base in ['T', 'C']:   return 'T/C'
        if base in ['TP', 'CP']: return 'TP/CP'
        if base == 'P':  return 'P'
        if base == 'PO': return 'PO'
        if base == 'O':  return 'O'
        if base == 'I':  return 'I'
        
        return 'Other'

    def is_left(self, ch_name):
        # Odd numbers are left
        nums = re.findall(r'\d+', ch_name)
        if nums:
            return int(nums[0]) % 2 != 0
        return False

    def is_right(self, ch_name):
        # Even numbers are right
        nums = re.findall(r'\d+', ch_name)
        if nums:
            return int(nums[0]) % 2 == 0
        return False

    # --- REFRESH LOGIC ---

    def refresh(self):
        """Builds the anatomically sorted grid."""
        # 1. Cleanup
        for i in reversed(range(self.grid_layout.count())): 
            w = self.grid_layout.itemAt(i).widget()
            if w: w.setParent(None)
        
        for i in reversed(range(self.row_layout.count())):
            w = self.row_layout.itemAt(i).widget()
            if w: w.setParent(None)
            
        self.checkboxes = {}

        if not self.session.current_data:
            return

        ch_names = self.session.current_data.ch_names
        
        # 2. Categorize Channels
        rows_order = ['Fp', 'AF', 'F', 'FT/FC', 'T/C', 'TP/CP', 'P', 'PO', 'O', 'I', 'EOG', 'EMG', 'Other']
        categorized = {r: [] for r in rows_order}
        
        for ch in ch_names:
            row = self.get_anatomical_row(ch)
            if row in categorized:
                categorized[row].append(ch)
            else:
                categorized['Other'].append(ch)

        # 3. Build UI Rows
        current_grid_row = 0
        
        for row_label in rows_order:
            channels = categorized[row_label]
            if not channels: continue
            
            # A. Add "Select Row" Button to top bar
            btn_row = QPushButton(row_label)
            # CHANGE: Connect to toggle_list instead of select_list
            btn_row.clicked.connect(lambda _, chs=channels: self.toggle_list(chs))
            
            self.row_layout.addWidget(btn_row)
            
            # B. Add Label to Grid
            lbl = QLabel(f"<b>{row_label}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.grid_layout.addWidget(lbl, current_grid_row, 0)
            
            # C. Add Checkboxes
            col_idx = 1
            MAX_COLS = 8 
            
            for ch in channels:
                chk = QCheckBox(ch)
                chk.setChecked(True) 
                chk.stateChanged.connect(self.update_session_selection)
                
                self.grid_layout.addWidget(chk, current_grid_row, col_idx)
                self.checkboxes[ch] = chk
                
                col_idx += 1
                if col_idx > MAX_COLS:
                    current_grid_row += 1
                    col_idx = 1
            
            current_grid_row += 1
            
        self.update_session_selection()

    # --- SELECTION SLOTS ---
    def toggle_list(self, ch_names):
        """
        Smart Toggle Logic:
        - If ALL are checked -> Uncheck all.
        - Otherwise (if some or none are checked) -> Check all.
        """
        # 1. Determine current state
        all_checked = True
        for ch in ch_names:
            if ch in self.checkboxes:
                if not self.checkboxes[ch].isChecked():
                    all_checked = False
                    break
        
        # 2. Toggle target state
        target_state = not all_checked  # If all checked, target is False. Else True.
        
        # 3. Apply
        # We block signals briefly to prevent firing 'update_session' N times
        self.blockSignals(True) 
        for ch in ch_names:
            if ch in self.checkboxes:
                self.checkboxes[ch].setChecked(target_state)
        self.blockSignals(False)
        
        # 4. Update session once at the end
        self.update_session_selection()
    
    def toggle_by_condition(self, condition_func):
        """
        Toggles channels matching a condition (Left, Right, Center).
        Logic: If ANY matching channel is unchecked -> Check ALL matching.
               Else (all matching are checked) -> Uncheck ALL matching.
        """
        # 1. Identify target channels
        target_chs = [ch for ch in self.checkboxes.keys() if condition_func(ch)]
        
        if not target_chs: return

        # 2. Determine current state of these channels
        all_checked = True
        for ch in target_chs:
            if not self.checkboxes[ch].isChecked():
                all_checked = False
                break
        
        # 3. Determine target state
        target_state = not all_checked
        
        # 4. Apply
        self.blockSignals(True)
        for ch in target_chs:
            self.checkboxes[ch].setChecked(target_state)
        self.blockSignals(False)
        
        self.update_session_selection()
        
    def update_session_selection(self):
        selected = [ch for ch, box in self.checkboxes.items() if box.isChecked()]
        self.session.selected_channels = selected 

    def select_all(self):
        for box in self.checkboxes.values(): box.setChecked(True)
    
    def select_none(self):
        for box in self.checkboxes.values(): box.setChecked(False)

    def select_list(self, ch_names):
        """Selects only the channels in the provided list (additive or exclusive?)
        Let's make it additive (check them, don't uncheck others).
        To make it exclusive, uncheck all first. Let's do Toggle logic?
        Standard UI: Buttons usually 'Select' (Check).
        """
        for ch in ch_names:
            if ch in self.checkboxes:
                self.checkboxes[ch].setChecked(True)

    def select_by_condition(self, condition_func):
        """Generic selector."""
        for ch, box in self.checkboxes.items():
            if condition_func(ch):
                box.setChecked(True)
            else:
                box.setChecked(False) # Strict selection

    def select_eeg(self):
        if not self.session.current_data: return
        # Uncheck all first
        self.select_none()
        try:
            picks = mne.pick_types(self.session.current_data.info, eeg=True)
            eeg_names = [self.session.current_data.ch_names[i] for i in picks]
            self.select_list(eeg_names)
        except:
            pass