"""Application settings, loaded from environment variables and backend/.env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Camera -----------------------------------------------------------
    # "stream" (ESP32-CAM MJPEG), "webcam" / "webcam:<index>", "file:<path>", or "none".
    cam_source: str = "stream"
    cam_stream_url: str = "http://192.168.4.2:81/stream"
    cam_capture_url: str = "http://192.168.4.2/capture"

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["anthropic", "qwen", "mock"] = "anthropic"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Server-side refusal fallback ("fallbacks": "default") on the Claude API.
    llm_fallbacks: bool = True
    llm_timeout_s: float = 30.0
    min_confidence: float = 0.55
    # Local Qwen3-VL (LLM_PROVIDER=qwen; run inside the cooking-vlm conda env).
    qwen_model: str = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    qwen_max_tokens: int = 512

    # --- Speech -------------------------------------------------------------
    stt_provider: Literal["whisper", "mock"] = "whisper"
    whisper_model: str = "base"
    mock_transcript: str = ""
    tts_provider: Literal["edge", "beep"] = "edge"
    tts_voice: str = "en-US-AriaNeural"

    # --- Gestures -----------------------------------------------------------
    gesture_enabled: bool = True
    gesture_model_path: Path = MODELS_DIR / "gesture_recognizer.task"
    gesture_fps: float = 10.0
    gesture_hold_frames: int = 5
    gesture_min_score: float = 0.6
    gesture_cooldown_s: float = 3.0
    gesture_input_width: int = 320

    # --- Orchestrator -------------------------------------------------------
    auto_start_timer: bool = True

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
