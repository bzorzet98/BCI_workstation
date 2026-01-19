from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (QComboBox)
class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().viewport().installEventFilter(self)
        self.model().itemChanged.connect(self._handle_item_changed)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().installEventFilter(self) # Catch focus events

    def _handle_item_changed(self, item):
        checked = self.get_checked_items()
        text = ", ".join(checked) if checked else "None selected"
        self.setEditText(text)

    def eventFilter(self, obj, event):
        # Prevent the popup from closing when clicking an item checkbox
        if obj == self.view().viewport() and event.type() == event.Type.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            item = self.model().itemFromIndex(index)
            if item:
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
            return True # Eat the event so the popup stays open
        return super().eventFilter(obj, event)

    def hidePopup(self):
        # Only hide if the mouse is not over the viewport (user clicked outside)
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