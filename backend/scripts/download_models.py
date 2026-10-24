"""Download the MediaPipe gesture recognizer model into backend/models/.

    python scripts/download_models.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS = {
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
}


def download(name: str, url: str, force: bool) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[ok] {dest} already present ({dest.stat().st_size // 1024} KiB)")
        return
    print(f"[..] downloading {name} from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.replace(dest)
    print(f"[ok] saved {dest} ({dest.stat().st_size // 1024} KiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODELS.items():
        try:
            download(name, url, args.force)
        except Exception as exc:  # noqa: BLE001
            print(f"[!!] failed to download {name}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
