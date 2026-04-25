"""
SafeSight AI - Video Stream Utility
Manages camera/video capture with threading for non-blocking frame reads.
"""

import cv2
import threading
import time
import numpy as np
from typing import Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class VideoStream:
    """
    Thread-safe video stream wrapper for OpenCV.
    Supports RTSP, HTTP streams, local files, and webcams.
    """

    def __init__(
        self,
        source,
        camera_id: str = "cam_01",
        target_fps: float = 30.0,
        reconnect_delay: float = 3.0,
        max_reconnect_attempts: int = 10,
        frame_buffer_size: int = 2,
    ):
        self.source = source
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps_actual = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0

    def start(self):
        """Start the background capture thread."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait for first frame
        timeout = 5.0
        start = time.time()
        while self._frame is None and time.time() - start < timeout:
            time.sleep(0.05)
        logger.info(f"VideoStream {self.camera_id} started from {self.source}")
        return self

    def stop(self):
        """Stop the capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        logger.info(f"VideoStream {self.camera_id} stopped.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame (non-blocking)."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def _capture_loop(self):
        reconnect_attempts = 0
        while self._running:
            if not self._open_capture():
                reconnect_attempts += 1
                if reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for {self.camera_id}. Stopping.")
                    self._running = False
                    break
                time.sleep(self.reconnect_delay)
                continue

            reconnect_attempts = 0
            frame_interval = 1.0 / self.target_fps

            while self._running:
                t0 = time.time()
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    logger.warning(f"Stream {self.camera_id} frame read failed. Reconnecting...")
                    break

                with self._lock:
                    self._frame = frame
                    self._frame_count += 1

                # FPS tracking
                self._fps_frame_count += 1
                elapsed = time.time() - self._last_fps_time
                if elapsed >= 2.0:
                    self._fps_actual = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._last_fps_time = time.time()

                # Pace to target FPS
                sleep_time = frame_interval - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if self._cap:
                self._cap.release()
                self._cap = None

    def _open_capture(self) -> bool:
        try:
            # Try integer device index
            source = int(self.source)
        except (ValueError, TypeError):
            source = self.source

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.warning(f"Cannot open stream: {self.source}")
            return False
        return True

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._running

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Returns (width, height) of the stream."""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 640, 480

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()
