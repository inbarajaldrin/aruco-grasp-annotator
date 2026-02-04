"""Session state management for ArUco Annotator."""

from typing import Any, Optional

from .marker import MarkerData


class SessionState:
    """Global session state for the annotator application."""

    def __init__(self) -> None:
        self.mesh: Optional[Any] = None
        self.mesh_info: Optional[dict] = None
        self.markers: dict[int, MarkerData] = {}
        self.next_marker_id: int = 0
        self.cad_object_info: Optional[dict] = None
        self.current_file: Optional[str] = None

    def reset(self) -> None:
        """Reset session state."""
        self.mesh = None
        self.mesh_info = None
        self.markers = {}
        self.next_marker_id = 0
        self.cad_object_info = None
        self.current_file = None

    def clear_markers(self) -> None:
        """Clear all markers."""
        self.markers = {}
        self.next_marker_id = 0

    def add_marker(self, marker: MarkerData) -> int:
        """Add a marker and return its internal ID."""
        internal_id = self.next_marker_id
        self.markers[internal_id] = marker
        self.next_marker_id += 1
        return internal_id

    def get_marker(self, marker_id: int) -> Optional[MarkerData]:
        """Get a marker by ID."""
        return self.markers.get(marker_id)

    def delete_marker(self, marker_id: int) -> bool:
        """Delete a marker by ID. Returns True if deleted."""
        if marker_id in self.markers:
            del self.markers[marker_id]
            return True
        return False

    def has_model(self) -> bool:
        """Check if a model is loaded."""
        return self.mesh is not None

    def has_cad_info(self) -> bool:
        """Check if CAD info is available."""
        return self.cad_object_info is not None


# Global session instance
session_state = SessionState()
