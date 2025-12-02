"""Merge view placeholder for shard merging operations."""

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class MergeView(QWidget):
    """Placeholder view for merging FBIN shards.
    
    This functionality will be implemented in a future milestone.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        
        label = QLabel(
            "Shard Merge Tool\n\n"
            "This feature allows you to merge multiple FBIN shard files "
            "into a single contiguous file.\n\n"
            "Coming in a future milestone."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()
