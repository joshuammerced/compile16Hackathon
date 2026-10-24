"""Pretend to be the ESP32-CAM: serve a webcam, looping video or still image as MJPEG.

    python scripts/fake_camera.py                      # webcam 0
    python scripts/fake_camera.py --source food.mp4    # loop a video
    python scripts/fake_camera.py --source egg.jpg     # a still image
    python scripts/fake_camera.py --port 8081          # if :81 needs admin rights

Endpoints (same paths as the Espressif camera web server):
    http://localhost:81/stream    multipart MJPEG
    http://localhost:81/capture   single JPEG
Then run the backend with CAM_STREAM_URL=http://localhost:81/stream
and CAM_CAPTURE_URL=http://localhost:81/capture.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

BOUNDARY = "123456789000000000000987654321"  # same boundary string the ESP32 firmware uses
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameSource(threading.Thread):
    """Reads frames in the background and keeps the latest JPEG."""

    def __init__(self, source: str, fps: float, width: int):
        super().__init__(daemon=True)
        self.source, self.period, self.width = source, 1.0 / fps, width
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.seq = 0

    def _encode(self, image) -> None:
        h, w = image.shape[:2]
        if w > self.width:
            image = cv2.resize(image, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg, self.seq = buf.tobytes(), self.seq + 1

    def run(self) -> None:
        if Path(self.source).suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(self.source)
            if image is None:
                raise SystemExit(f"cannot read image {self.source}")
            while True:
                self._encode(image)
                time.sleep(self.period)
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source {self.source}")
        while True:
            ok, image = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop videos
                time.sleep(0.05)
                continue
            self._encode(image)
            time.sleep(self.period)


def make_handler(frames: FrameSource):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter than the default
            print(f"[fake_camera] {self.address_string()} {fmt % args}")

        def _latest(self) -> tuple[bytes | None, int]:
            with frames.lock:
                return frames.jpeg, frames.seq

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/capture"):
                jpeg, _ = self._latest()
                if jpeg is None:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
                self.end_headers()
                last = -1
                try:
                    while True:
                        jpeg, seq = self._latest()
                        if jpeg is None or seq == last:
                            time.sleep(0.01)
                            continue
                        last = seq
                        self.wfile.write(
                            f"\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                            + jpeg
                        )
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            else:
                body = b'<html><body><img src="/stream"></body></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index, video file or image (default: 0)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=81)
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--width", type=int, default=320, help="resize to this width (QVGA = 320)")
    args = parser.parse_args()

    frames = FrameSource(args.source, args.fps, args.width)
    frames.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(frames))
    server.daemon_threads = True
    print(f"[fake_camera] serving {args.source} at http://localhost:{args.port}/stream and /capture")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
