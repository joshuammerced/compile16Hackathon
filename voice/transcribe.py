"""Transcribe a WAV file to text with Whisper base (faster-whisper, CPU int8).

Usage:
    conda activate cooking-vlm
    python voice/transcribe.py record_out.wav
    python voice/transcribe.py record_out.wav --model small --language en
"""

from __future__ import annotations

import argparse
import sys
import time

from faster_whisper import WhisperModel


def main() -> int:
    parser = argparse.ArgumentParser(description="Speech-to-text for a WAV file using Whisper.")
    parser.add_argument("wav", help="path to the audio file (any sample rate; resampled to 16 kHz)")
    parser.add_argument("--model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--language", default="en", help="spoken language, or 'auto' to detect")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    load_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        args.wav,
        beam_size=1,
        language=None if args.language == "auto" else args.language,
        vad_filter=True,
    )
    segments = list(segments)  # transcription runs lazily while iterating
    infer_s = time.perf_counter() - t0

    for seg in segments:
        print(f"[{seg.start:6.2f}s -> {seg.end:6.2f}s] {seg.text.strip()}")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"\nTranscript: {text or '(no speech detected)'}")
    print(
        f"model={args.model} language={info.language} audio={info.duration:.1f}s "
        f"load={load_s:.1f}s transcribe={infer_s:.2f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
