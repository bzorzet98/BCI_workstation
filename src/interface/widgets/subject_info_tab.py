import pprint
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit,  QGroupBox

class SubjectInfoTab(QWidget):
    """
    Displays specific technical metadata for the selected subject (Extra Info).
    """
    def __init__(self, session, style_kwargs=None):
        super().__init__()
        self.session = session
        self.style_kwargs = style_kwargs or {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        self.group = QGroupBox("Subject Technical Information")
        v_box = QVBoxLayout(self.group)
        
        self.info_display = QTextEdit()
        self.info_display.setReadOnly(True)
        
        v_box.addWidget(self.info_display)
        layout.addWidget(self.group)

    def refresh(self):
        """Updates the display safely."""
        try:
            if hasattr(self.session, 'extra_info') and self.session.extra_info:
                # Use pprint.pformat instead of json.dumps. 
                # It is MUCH more robust with NumPy types and MNE objects.
                formatted_text = pprint.pformat(self.session.extra_info, indent=4, width=80)
                self.info_display.setPlainText(formatted_text)
            else:
                self.info_display.setPlainText("No extra information available for this subject.")
        except Exception as e:
            self.info_display.setPlainText(f"Error displaying metadata: {str(e)}")