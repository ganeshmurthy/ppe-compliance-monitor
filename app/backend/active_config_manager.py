"""Active configuration state manager with pub-sub pattern for SSE notifications."""

import threading
from typing import Callable, Optional

from logger import get_logger

log = get_logger(__name__)


class ActiveConfigManager:
    """Manages the currently active video configuration and notifies subscribers of changes.

    This enables Server-Sent Events (SSE) clients to be notified when any user
    switches to a different video source, allowing all UIs to update their
    highlighted thumbnail in real-time.
    """

    def __init__(self):
        """Initialize the manager with no active config."""
        self._active_config_id: Optional[int] = None
        self._video_source: Optional[str] = None
        self._subscribers: list[Callable[[int, str], None]] = []
        self._lock = threading.Lock()

    def set_active_config(self, config_id: int, video_source: str) -> None:
        """Set the active configuration and notify all subscribers.

        Args:
            config_id: Database ID of the configuration
            video_source: Path or URL of the video source

        All registered subscriber callbacks will be invoked with the new config_id
        and video_source.
        """
        with self._lock:
            self._active_config_id = config_id
            self._video_source = video_source
            # Copy subscribers list to avoid holding lock during callbacks
            subscribers = list(self._subscribers)

        # Notify all subscribers (outside the lock to avoid deadlock)
        for subscriber in subscribers:
            try:
                subscriber(config_id, video_source)
            except Exception as e:
                # Log but don't propagate subscriber errors
                log.exception(f"Subscriber callback failed: {e}")

    def get_active_config(self) -> tuple[Optional[int], Optional[str]]:
        """Get the current active configuration.

        Returns:
            tuple: (config_id, video_source) or (None, None) if no active config
        """
        with self._lock:
            return self._active_config_id, self._video_source

    def subscribe(self, callback: Callable[[int, str], None]) -> None:
        """Subscribe to active config change notifications.

        Args:
            callback: Function to call when active config changes.
                     Signature: callback(config_id: int, video_source: str)

        The callback will be invoked immediately with the current config (if any),
        then again whenever the config changes.
        """
        with self._lock:
            self._subscribers.append(callback)
            current_config_id = self._active_config_id
            current_video_source = self._video_source

        # Send current state to new subscriber (if there is one)
        if current_config_id is not None and current_video_source is not None:
            try:
                callback(current_config_id, current_video_source)
            except Exception as e:
                log.exception(f"Subscriber callback failed on subscribe: {e}")

    def unsubscribe(self, callback: Callable[[int, str], None]) -> None:
        """Unsubscribe from active config change notifications.

        Args:
            callback: The callback function to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
