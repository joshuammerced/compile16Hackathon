"""Adapter for the team's local Qwen3-VL-4B model in LLM/inference.py (MLX, Apple Silicon).

The module is imported from its file path (VLM_INFERENCE_PATH) so LLM/ stays the single source
of truth for the model id and COOKING_PROMPT. ``CookingVLM.assess()`` takes an image path, so
each frame is written to one reused temp file; only one inference thread ever touches it.
"""

from __future__ import annotations

import importlib.util
import logging
import tempfile
import time
from pathlib import Path
from types import ModuleType

from .base import VisionProvider

log = logging.getLogger(__name__)


def load_inference_module(path: Path) -> ModuleType:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"vision model code not found: {path} (set VLM_INFERENCE_PATH)")
    spec = importlib.util.spec_from_file_location("cooking_inference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QwenVLMProvider(VisionProvider):
    name = "qwen3-vl"

    def __init__(self, inference_path: Path, model_id: str | None = None, max_tokens: int = 200):
        self.inference_path = Path(inference_path)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._vlm = None
        self._tmp = Path(tempfile.mkdtemp(prefix="cooking-vlm-")) / "frame.jpg"

    def load(self) -> None:
        module = load_inference_module(self.inference_path)
        self.model_id = self.model_id or module.MODEL_ID
        t0 = time.perf_counter()
        # Raises module.ModelLoadError on machines without MLX (anything but Apple Silicon).
        self._vlm = module.CookingVLM(self.model_id)
        log.info("loaded %s in %.1fs", self.model_id, time.perf_counter() - t0)

    def assess(self, jpeg: bytes) -> str:
        if self._vlm is None:
            raise RuntimeError("model not loaded")
        self._tmp.write_bytes(jpeg)
        result = self._vlm.assess(self._tmp, max_tokens=self.max_tokens)
        return result.text
