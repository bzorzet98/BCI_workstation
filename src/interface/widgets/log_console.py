import sys
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import pyqtSignal, QObject

class StreamRedirector(QObject):
    """Emits text written to stdout as a signal."""
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class LogConsole(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        # Setup redirection
        self.redirector = StreamRedirector()
        self.redirector.textWritten.connect(self.append_text)
        
        # Redirect stdout/stderr
        sys.stdout = self.redirector
        sys.stderr = self.redirector

    def append_text(self, text):
        self.moveCursor(self.textCursor().MoveOperation.End)
        self.insertPlainText(text)
        self.moveCursor(self.textCursor().MoveOperation.End)