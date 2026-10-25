"""Application settings, loaded from environment variables and backend/.env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"
REPO_DIR = BACKEND_DIR.parent


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

    # --- Vision model (LLM/inference.py, Qwen3-VL-4B on MLX) ---------------------
    # qwen: the team's local model; mock: canned observations (no model, any OS)
    vision_provider: Literal["qwen", "mock"] = "qwen"
    vlm_inference_path: Path = REPO_DIR / "LLM" / "inference.py"
    vlm_model_id: str | None = None  # default: MODEL_ID from inference.py
    vlm_max_tokens: int = 200

    # --- Continuous monitoring ------------------------------------------------
    monitor_interval_s: float = 4.0  # how often to look during a session (one inference in flight max)
    frame_max_age_s: float = 2.0  # never analyze a frame older than this
    result_max_age_s: float = 20.0  # drop a result whose frame is older than this when it arrives
    overcook_confirm: int = 2  # consecutive "burnt" observations before warning
    overcook_repeat_s: float = 60.0
    looks_done_confirm: int = 2
    camera_lost_after_s: float = 20.0

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
    # Typical cook times are suggestions; by default the cook confirms with a thumbs-up.
    auto_start_timer: bool = False
    state_file: Path | None = BACKEND_DIR / "data" / "session.json"  # empty to disable persistence

    # --- Server -------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @field_validator("state_file", "vlm_model_id", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("vlm_inference_path", mode="before")
    @classmethod
    def _empty_is_default(cls, v):
        return REPO_DIR / "LLM" / "inference.py" if isinstance(v, str) and not v.strip() else v

    @field_validator("state_file", "vlm_inference_path", "gesture_model_path")
    @classmethod
    def _relative_to_backend(cls, v: Path | None) -> Path | None:
        return v if v is None or v.is_absolute() else BACKEND_DIR / v


@lru_cache
def get_settings() -> Settings:
    return Settings()
