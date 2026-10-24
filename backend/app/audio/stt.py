"""Speech-to-text: faster-whisper on CPU, plus a mock."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod

from .wav import decode_audio

log = logging.getLogger(__name__)


class STTProvider(ABC):
    name = "base"

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str: ...


class MockSTT(STTProvider):
    """Returns a fixed transcript (MOCK_TRANSCRIPT), or the X-Transcript header if the route passes one."""

    name = "mock"

    def __init__(self, transcript: str = ""):
        self.transcript = transcript

    async def transcribe(self, audio: bytes) -> str:
        return self.transcript


class WhisperSTT(STTProvider):
    name = "whisper"

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                t0 = time.perf_counter()
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                log.info("loaded faster-whisper %s in %.1fs", self.model_size, time.perf_counter() - t0)
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        model = self._load()
        samples = decode_audio(audio)
        t0 = time.perf_counter()
        segments, _info = model.transcribe(samples, language="en", beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log.info("whisper transcribed %.1fs of audio in %.2fs: %r", len(samples) / 16000, time.perf_counter() - t0, text)
        return text

    def warm_up(self) -> None:
        self._load()

    async def transcribe(self, audio: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio)


def create_stt(provider: str, whisper_model: str, mock_transcript: str) -> STTProvider:
    if provider == "whisper":
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            log.warning("faster-whisper not installed; using mock STT")
            return MockSTT(mock_transcript)
        return WhisperSTT(whisper_model)
    return MockSTT(mock_transcript)
