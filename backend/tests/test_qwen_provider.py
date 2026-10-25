"""The adapter around LLM/inference.py, tested against a stub with the same API (MLX only runs
on Apple Silicon, so the real model can't run in CI)."""

import textwrap

import cv2
import numpy as np
import pytest

from app.llm.qwen_provider import QwenVLMProvider, load_inference_module
from app.config import REPO_DIR

STUB = textwrap.dedent(
    '''
    from pathlib import Path
    MODEL_ID = "stub/model"
    class ModelLoadError(RuntimeError): ...
    class Result:
        def __init__(self, text): self.text = text
    class CookingVLM:
        instances = 0
        def __init__(self, model_id=MODEL_ID):
            CookingVLM.instances += 1
            self.model_id = model_id
            self.seen = []
        def assess(self, image_path, max_tokens=200):
            data = Path(image_path).read_bytes()
            assert data[:2] == b"\\xff\\xd8", "expected a JPEG file"
            self.seen.append(max_tokens)
            return Result("Food Identified: Toast\\nCooking Status: APPEARS_COOKED\\nVisual Evidence: Golden.")
    '''
)


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))[1].tobytes()


def test_adapter_loads_once_and_passes_frames_as_files(tmp_path):
    path = tmp_path / "inference.py"
    path.write_text(STUB)
    p = QwenVLMProvider(path, max_tokens=123)
    p.load()
    assert p.model_id == "stub/model"
    for _ in range(3):
        assert "APPEARS_COOKED" in p.assess(jpeg())
    assert p._vlm.seen == [123, 123, 123]
    assert type(p._vlm).instances == 1  # model kept in memory, not reloaded per frame


def test_missing_inference_file_fails_load(tmp_path):
    with pytest.raises(FileNotFoundError):
        QwenVLMProvider(tmp_path / "nope.py").load()


def test_real_inference_module_imports_and_matches_contract():
    """The team's file must expose what the adapter relies on (import only; no model load)."""
    module = load_inference_module(REPO_DIR / "LLM" / "inference.py")
    assert isinstance(module.MODEL_ID, str) and hasattr(module, "CookingVLM")
    for field in ("Food Identified:", "Cooking Status:", "Visual Evidence:", "Recommendation:"):
        assert field in module.COOKING_PROMPT
    for status in ("APPEARS_COOKED", "APPEARS_UNDERCOOKED", "POSSIBLY_OVERCOOKED_OR_BURNT", "CANNOT_DETERMINE"):
        assert status in module.COOKING_PROMPT
