import mne
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QComboBox, 
                             QSpinBox, QListWidget, QPushButton, 
                             QFormLayout, QAbstractItemView, 
                             QApplication, QMessageBox, QHBoxLayout, QGroupBox, QLabel)
from PyQt6.QtCore import pyqtSignal, Qt

class WorkspaceTab(QWidget):
    """
    WorkspaceTab allows users to interactively configure and load EEG data.
    It supports multi-context loading, session selection, and extra metadata retrieval.
    """
    dataLoaded = pyqtSignal()

    def __init__(self, session, data_manager, style_kwargs=None):
        super().__init__()
        self.session = session
        self.data_manager = data_manager
        self.style_kwargs = style_kwargs or {}
        
        self._init_ui()

    def _init_ui(self):
        """Builds a compact, top-aligned configuration layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # 1. PARAMETERS GROUP
        self.config_group = QGroupBox("Data Configuration")
        form = QFormLayout(self.config_group)
        form.setContentsMargins(20, 25, 20, 20)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Dataset Selector
        self.cb_dataset = QComboBox()
        self.cb_dataset.addItems(self.data_manager.get_available_datasets())
        self.cb_dataset.currentTextChanged.connect(self._on_dataset_change)

        # Subject Selector
        self.sb_subject = QSpinBox()
        self.sb_subject.setRange(1, 300)

        # NEW: Session Selector (Interactive)
        self.cb_session = QComboBox()
        
        # Contexts Multi-Selector
        self.list_contexts = QListWidget()
        self.list_contexts.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.list_contexts.setFixedHeight(180) 

        # Add rows to form
        form.addRow("Dataset Source:", self.cb_dataset)
        form.addRow("Subject ID:", self.sb_subject)
        form.addRow("Target Session:", self.cb_session) # <--- New row
        form.addRow("Select Signals (Contexts):", self.list_contexts)

        layout.addWidget(self.config_group)

        # 2. ACTION BAR
        h_btn_layout = QHBoxLayout()
        h_btn_layout.addStretch() 
        
        self.btn_load = QPushButton("BUILD WORKSPACE")
        self.btn_load.setFixedSize(220, 40) 
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.clicked.connect(self._execute_load)
        
        h_btn_layout.addWidget(self.btn_load)
        layout.addLayout(h_btn_layout)

        # 3. FILLER STRETCH
        layout.addStretch()

        # Initial trigger to populate the first dataset's info
        if self.cb_dataset.count() > 0:
            self._on_dataset_change(self.cb_dataset.currentText())

    # --- LOGIC ---

    def _on_dataset_change(self, name: str):
        """Updates subject range, available sessions, and contexts when dataset changes."""
        try:
            self.data_manager.load_dataset_class(name)
            meta = self.data_manager.get_metadata()
            
            if meta:
                # 1. Update Contexts List
                self.list_contexts.clear()
                if 'contexts' in meta: 
                    self.list_contexts.addItems(meta['contexts'])
                
                # 2. Update Subject Range
                if 'subjects' in meta and meta['subjects']:
                    self.sb_subject.setRange(min(meta['subjects']), max(meta['subjects']))
                    self.sb_subject.setValue(min(meta['subjects']))
                
                # 3. Update Sessions Selector
                self.cb_session.clear()
                if 'sessions' in meta:
                    self.cb_session.addItems(meta['sessions'])
                else:
                    # Fallback if dataset class doesn't define sessions explicitly
                    self.cb_session.addItem("session_1")

        except Exception as e:
            print(f"[WorkspaceTab] Metadata update error: {e}")

    def _execute_load(self):
        """Retrieves user selection and initiates the data load via DataManager."""
        selected_contexts = [item.text() for item in self.list_contexts.selectedItems()]
        
        if not selected_contexts:
            QMessageBox.warning(self, "Selection Required", "Please select at least one context (signal).")
            return

        # Show waiting cursor (feedback for the user)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            subject = self.sb_subject.value()
            session_id = self.cb_session.currentText()
            dataset_name = self.cb_dataset.currentText()

            # 1. Call DataManager (now returns data dict AND extra_info)
            data_dict, extra_info = self.data_manager.load_data(
                subject_id=subject, 
                contexts=selected_contexts, 
                session_id=session_id
            )
            
            if data_dict:
                # 2. Initialize the SessionState workspace
                self.session.load_data(data_dict, dataset_name, subject, extra_info)
                
                # 3. Notify the Main Window to update Focus Switcher and Tabs
                self.dataLoaded.emit()
            else:
                QMessageBox.critical(self, "Load Error", "No signals were found for this subject/session.")
        
        except Exception as e:
            QMessageBox.critical(self, "Critical Failure", f"Error building workspace: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()