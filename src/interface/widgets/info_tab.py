import pprint
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGroupBox, QTextEdit)

class DatasetInfoTab(QWidget):
    def __init__(self, session, data_manager):
        super().__init__()
        self.session = session
        self.data_manager = data_manager
        self._init_ui()
    
    def _init_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
    
        # 1. Dataset General Info
        self.grp_general = QGroupBox("Dataset Technical Details")
        self.lbl_general = QLabel("No dataset loaded.")
        self.lbl_general.setWordWrap(True)
        self.lbl_general.setStyleSheet("font-size: 14px; padding: 10px;")
        v1 = QVBoxLayout()
        v1.addWidget(self.lbl_general)
        self.grp_general.setLayout(v1)
        self.layout.addWidget(self.grp_general)
        
        # 2. Subject Extra Info (Scrollable JSON)
        self.grp_extra = QGroupBox("Subject / Run Specific Metadata")
        self.txt_extra = QTextEdit()
        self.txt_extra.setReadOnly(True)
        self.txt_extra.setPlaceholderText("No extra info available.")
        self.txt_extra.setStyleSheet("font-family: Consolas; font-size: 12px;")
        v2 = QVBoxLayout()
        v2.addWidget(self.txt_extra)
        self.grp_extra.setLayout(v2)
        self.layout.addWidget(self.grp_extra)

    def refresh(self):
        """Called when data changes."""
        ds = self.data_manager.active_dataset_instance
        
        # A. Update General Info
        if ds:
            txt = (f"<b>Code:</b> {getattr(ds, 'code', 'N/A')}<br>"
                   f"<b>Montage:</b> {getattr(ds, 'standard_montage', 'N/A')}<br>"
                   f"<b>Sampling Rate:</b> {getattr(ds, 'sfreq', 'N/A')} Hz<br>"
                   f"<b>Total Subjects:</b> {len(ds.get_subjects())}<br>"
                   f"<b>Event Map:</b> {getattr(ds, 'event_id', 'N/A')}")
            self.lbl_general.setText(txt)
        else:
            self.lbl_general.setText("No dataset loaded.")
        
        # B. Update Extra Info
        if hasattr(self.session, 'extra_info') and self.session.extra_info:
            formatted_text = pprint.pformat(self.session.extra_info, indent=2)
            self.txt_extra.setText(formatted_text)
        else:
            self.txt_extra.clear()