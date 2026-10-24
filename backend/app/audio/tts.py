"""Text-to-speech to 16 kHz / 16-bit / mono WAV, with an in-memory clip cache.

Real engine: edge-tts (Microsoft Edge online voices; needs internet, no API key). Its MP3
output is decoded with soundfile (libsndfile has MP3 support, so no ffmpeg is needed) and
resampled to 16 kHz. If it fails for any reason the caller gets a beep instead, so the device
always has something playable.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

from .wav import SAMPLE_RATE, decode_audio, encode_wav

log = logging.getLogger(__name__)


def beep(kind: str = "info") -> bytes:
    """Synthesize a short tone pattern: info (1 beep), done (3 rising notes), warning (4 fast
    high beeps, for overcooking), error (low double)."""
    patterns = {
        "info": [(880, 0.18)],
        "warning": [(1400, 0.15), (0, 0.06), (1400, 0.15), (0, 0.06), (1400, 0.15), (0, 0.06), (1400, 0.15)],
        "done": [(880, 0.2), (0, 0.08), (1175, 0.2), (0, 0.08), (1568, 0.35)],
        "error": [(330, 0.2), (0, 0.08), (262, 0.3)],
    }
    chunks = []
    for freq, dur in patterns.get(kind, patterns["info"]):
        t = np.arange(int(SAMPLE_RATE * dur)) / SAMPLE_RATE
        tone = 0.5 * np.sin(2 * np.pi * freq * t) if freq else np.zeros_like(t)
        # 10 ms fade in/out to avoid clicks.
        fade = min(len(t) // 2, int(0.01 * SAMPLE_RATE))
        if fade:
            env = np.ones_like(t)
            env[:fade] = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            tone *= env
        chunks.append(tone)
    return encode_wav(np.concatenate(chunks).astype(np.float32))


class TTSProvider(ABC):
    name = "base"

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Return a 16 kHz / 16-bit / mono WAV."""


class BeepTTS(TTSProvider):
    name = "beep"

    async def synthesize(self, text: str) -> bytes:
        return beep("info")


class EdgeTTS(TTSProvider):
    name = "edge"

    def __init__(self, voice: str = "en-US-AriaNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        import edge_tts

        mp3 = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        if not mp3:
            raise RuntimeError("edge-tts returned no audio")
        return encode_wav(decode_audio(bytes(mp3)))


class AudioClipStore:
    """Keeps the last ``max_clips`` WAV clips in memory, keyed by a short id.

    Pinned clips (audio of unacknowledged alerts) are never evicted, so an alert's audio stays
    fetchable for as long as the alert is pending.
    """

    def __init__(self, max_clips: int = 10):
        self.max_clips = max_clips
        self._clips: OrderedDict[str, bytes] = OrderedDict()
        self._pinned: set[str] = set()
        self._lock = threading.Lock()

    def put(self, wav: bytes, pin: bool = False) -> str:
        with self._lock:
            clip_id = secrets.token_hex(3)
            while clip_id in self._clips:
                clip_id = secrets.token_hex(3)
            self._clips[clip_id] = wav
            if pin:
                self._pinned.add(clip_id)
            unpinned = [k for k in self._clips if k not in self._pinned]
            for k in unpinned[: max(0, len(unpinned) - self.max_clips)]:
                del self._clips[k]
            return clip_id

    def unpin(self, clip_id: str | None) -> None:
        with self._lock:
            self._pinned.discard(clip_id)

    def get(self, clip_id: str) -> bytes | None:
        with self._lock:
            return self._clips.get(clip_id)


class Speaker:
    """TTS front-end: synthesize with the configured engine, fall back to a beep, store the clip."""

    def __init__(self, provider: TTSProvider, store: AudioClipStore | None = None):
        self.provider = provider
        self.store = store or AudioClipStore()

    async def speak(self, text: str, fallback_kind: str = "info", chime: str | None = None, pin: bool = False) -> str:
        """Synthesize ``text`` (optionally preceded by a beep pattern) and return the clip id."""
        t0 = time.perf_counter()
        if isinstance(self.provider, BeepTTS) or not text.strip():
            wav = beep(chime or fallback_kind)
        else:
            try:
                wav = await self.provider.synthesize(text)
                if chime:
                    gap = np.zeros(int(0.15 * SAMPLE_RATE), dtype=np.float32)
                    wav = encode_wav(np.concatenate([decode_audio(beep(chime)), gap, decode_audio(wav)]))
            except Exception as exc:  # noqa: BLE001 - always produce playable audio
                log.warning("TTS %s failed (%s); using beep", self.provider.name, exc)
                wav = beep(chime or fallback_kind)
        clip_id = self.store.put(wav, pin=pin)
        log.info("TTS %s: %d bytes in %.2fs -> %s: %r", self.provider.name, len(wav), time.perf_counter() - t0, clip_id, text)
        return clip_id


def create_tts(provider: str, voice: str) -> TTSProvider:
    if provider == "edge":
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            log.warning("edge-tts not installed; using beep TTS")
            return BeepTTS()
        return EdgeTTS(voice)
    return BeepTTS()
