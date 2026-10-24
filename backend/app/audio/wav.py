"""WAV helpers: everything the device sends or receives is 16 kHz / 16-bit / mono PCM."""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000


def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Downmix to mono float32 and resample to 16 kHz."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        g = gcd(sample_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype(np.float32)
    return audio


def encode_wav(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode float samples (-1..1) as a 16 kHz, 16-bit, mono PCM WAV."""
    audio = np.clip(to_mono_16k(samples, sample_rate), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def decode_audio(data: bytes) -> np.ndarray:
    """Decode WAV/MP3/... bytes, or headerless 16 kHz s16le PCM, to mono float32 at 16 kHz."""
    if data[:4] != b"RIFF" and data[:3] != b"ID3" and data[:2] not in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        # Raw PCM straight from the BOX microphone, no header.
        pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
        return pcm.astype(np.float32) / 32768.0
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_mono_16k(samples, sr)
