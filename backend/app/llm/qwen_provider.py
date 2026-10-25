"""Local Qwen3-VL-4B provider (MLX-VLM on Apple Silicon).

Reuses ``CookingVLM`` from the repo's ``LLM/inference.py`` to load
``mlx-community/Qwen3-VL-4B-Instruct-4bit``. Must run in a Python that has mlx-vlm installed
(the ``cooking-vlm`` conda env). Structured output is requested in the prompt; the router
validates with Pydantic and retries once, as for every provider.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..schemas import decision_json_schema
from .base import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

LLM_DIR = Path(__file__).resolve().parents[3] / "LLM"

JSON_INSTRUCTIONS = """\

Respond with ONLY one JSON object (no markdown, no prose) that matches this JSON schema. \
Every property is required; use null where a field does not apply.
{schema}"""


class QwenProvider(LLMProvider):
    name = "qwen"

    def __init__(self, model: str | None = None, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self._schema = json.dumps(decision_json_schema())
        # MLX streams are per-thread: load and generate on the same single worker thread.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen")
        self._vlm = self._executor.submit(self._load, model).result()
        self.model = self._vlm.model_id

    @staticmethod
    def _load(model: str | None):
        if str(LLM_DIR) not in sys.path:
            sys.path.insert(0, str(LLM_DIR))
        from inference import MODEL_ID, CookingVLM

        model_id = model or MODEL_ID
        log.info("loading local VLM %s", model_id)
        return CookingVLM(model_id)

    def _prompt(self, request: LLMRequest) -> str:
        text = f"{request.system}\n{request.user_text}{JSON_INSTRUCTIONS.format(schema=self._schema)}"
        if request.retry_feedback:
            text += (
                f"\n\nYour previous answer was rejected by the validator: {request.retry_feedback}\n"
                "Return one corrected JSON object."
            )
        return text

    def _generate_sync(self, request: LLMRequest) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        vlm = self._vlm
        images = [Image.open(io.BytesIO(request.image_jpeg)).convert("RGB")] if request.image_jpeg else []
        prompt = apply_chat_template(vlm.processor, vlm.config, self._prompt(request), num_images=len(images))
        result = generate(
            vlm.model,
            vlm.processor,
            prompt,
            image=images or None,
            max_tokens=self.max_tokens,
            temperature=0.0,
            verbose=False,
        )
        log.debug(
            "qwen: %d prompt tokens, %d generated, %.1f tok/s",
            result.prompt_tokens,
            result.generation_tokens,
            result.generation_tps,
        )
        if result.finish_reason == "length":
            raise RuntimeError("model output truncated (max_tokens)")
        return result.text

    async def generate(self, request: LLMRequest) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._generate_sync, request)
