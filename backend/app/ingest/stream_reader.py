"""OpenCV ingest thread that keeps only the latest camera frame in RAM.

Sources (CAM_SOURCE):
  stream          ESP32-CAM MJPEG at CAM_STREAM_URL (http://<cam>:81/stream)
  webcam[:N]      local webcam, device index N (default 0)
  file:<path>     a video (looped) or a still image, for development
  none            no camera; get_snapshot() still tries CAM_CAPTURE_URL
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Frame:
    image: np.ndarray  # BGR
    timestamp: float  # time.monotonic() when captured
    seq: int

    def to_jpeg(self, quality: int = 85) -> bytes:
        ok, buf = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()


class StreamReader:
    def __init__(
        self,
        source: str,
        stream_url: str,
        capture_url: str | None = None,
        max_backoff_s: float = 10.0,
    ):
        self.source = source
        self.stream_url = stream_url
        self.capture_url = capture_url
        self.max_backoff_s = max_backoff_s

        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- public API -------------------------------------------------------------

    def start(self) -> None:
        if self.source == "none":
            log.info("camera source disabled (CAM_SOURCE=none)")
            return
        self._thread = threading.Thread(target=self._run, name="stream-reader", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("stream reader did not stop within %.1fs (blocked read)", timeout)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def fps(self) -> float:
        with self._lock:
            times = list(self._frame_times)
        if len(times) < 2 or times[-1] == times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    @property
    def last_frame_age_s(self) -> float | None:
        frame = self.latest()
        return None if frame is None else time.monotonic() - frame.timestamp

    def get_snapshot(self, max_age_s: float = 2.0) -> Frame | None:
        """Latest frame if fresh; otherwise fetch one still from CAM_CAPTURE_URL."""
        frame = self.latest()
        if frame is not None and time.monotonic() - frame.timestamp <= max_age_s:
            return frame
        snap = self._fetch_capture()
        return snap if snap is not None else frame

    # --- internals ---------------------------------------------------------------

    def _publish(self, image: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            self._seq += 1
            self._latest = Frame(image=image, timestamp=now, seq=self._seq)
            self._frame_times.append(now)

    def _fetch_capture(self) -> Frame | None:
        if not self.capture_url or self.source not in ("stream", "none"):
            return None
        try:
            with urllib.request.urlopen(self.capture_url, timeout=3) as resp:
                data = resp.read()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("capture returned non-JPEG data")
            log.info("fetched still from %s", self.capture_url)
            self._publish(image)
            return self.latest()
        except Exception as exc:  # noqa: BLE001 - camera may simply be offline
            log.warning("snapshot fetch from %s failed: %s", self.capture_url, exc)
            return None

    def _open(self) -> cv2.VideoCapture | None:
        if self.source.startswith("webcam"):
            _, _, idx = self.source.partition(":")
            cap = cv2.VideoCapture(int(idx or 0))
        elif self.source.startswith("file:"):
            cap = cv2.VideoCapture(self.source[5:])
        else:
            # FFmpeg backend handles multipart MJPEG over HTTP; bound open/read so a dead
            # camera can't wedge the thread forever.
            cap = cv2.VideoCapture(
                self.stream_url,
                cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
            )
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _describe(self) -> str:
        return self.stream_url if self.source == "stream" else self.source

    def _run(self) -> None:
        if self.source.startswith("file:") and Path(self.source[5:]).suffix.lower() in IMAGE_SUFFIXES:
            self._run_still_image(Path(self.source[5:]))
            return

        backoff = 0.5
        is_file = self.source.startswith("file:")
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._connected = False
                log.warning("camera %s unavailable, retrying in %.1fs", self._describe(), backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue

            log.info("camera connected: %s", self._describe())
            self._connected = True
            backoff = 0.5
            # Video files would otherwise be read as fast as possible; pace them to their FPS.
            file_period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0) if is_file else 0.0
            try:
                while not self._stop.is_set():
                    ok, image = cap.read()
                    if not ok or image is None:
                        if is_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                            continue
                        log.warning("camera stream dropped: %s", self._describe())
                        break
                    self._publish(image)
                    if file_period:
                        time.sleep(file_period)
            finally:
                cap.release()
                self._connected = False

    def _run_still_image(self, path: Path) -> None:
        image = cv2.imread(str(path))
        if image is None:
            log.error("could not read image %s", path)
            return
        log.info("camera source is still image %s", path)
        self._connected = True
        while not self._stop.is_set():
            self._publish(image)
            self._stop.wait(0.1)
        self._connected = False
