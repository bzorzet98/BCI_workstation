from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().viewport().installEventFilter(self)
        self.model().itemChanged.connect(self._handle_item_changed)
        
        # Enable the text area but make it read-only
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().installEventFilter(self) # Necessary to catch clicks on the text area
        
        # Aesthetic: change cursor to pointing hand to show it's clickable
        self.lineEdit().setCursor(Qt.CursorShape.PointingHandCursor)

    def _handle_item_changed(self, item):
        checked = self.get_checked_items()
        text = ", ".join(checked) if checked else "None selected"
        self.setEditText(text)

    def eventFilter(self, obj, event):
        # 1. NEW: Logic to open popup when clicking anywhere on the text area (lineEdit)
        if obj == self.lineEdit() and event.type() == event.Type.MouseButtonRelease:
            if not self.view().isVisible():
                self.showPopup()
            return True

        # 2. EXISTING: Prevent closing when clicking a checkbox in the list
        if obj == self.view().viewport() and event.type() == event.Type.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            item = self.model().itemFromIndex(index)
            if item:
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
            return True 
            
        return super().eventFilter(obj, event)

    def hidePopup(self):
        # Only hide if the mouse is not over the viewport (user clicked outside the list)
        if not self.view().viewport().underMouse():
            super().hidePopup()

    def get_checked_items(self):
        items = []
        for i in range(self.count()):
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                items.append(self.model().item(i).text())
        return items

    def addItem(self, text):
        super().addItem(text)
        item = self.model().item(self.count() - 1)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Unchecked)
        
    def set_item_checked(self, text, state=True):
        """Helper to check an item by its text name."""
        for i in range(self.count()):
            if self.itemText(i) == text:
                st = Qt.CheckState.Checked if state else Qt.CheckState.Unchecked
                self.model().item(i).setCheckState(st)
                break