"""Local Qwen3-VL-4B (MLX) cooking-state inference for the Elderly Cooking Assistant.

Phase 1: run a single image through the vision-language model and print the
assessment. `CookingVLM` is structured so it can later be held as a singleton
inside a FastAPI service (load once, call `assess()` per image).
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "images.jpeg"
MAX_TOKENS = 200
TEMPERATURE = 0.0

COOKING_PROMPT = """You are a visual cooking assistant designed to help elderly users monitor food preparation.

Carefully examine the attached image and determine what can reasonably be inferred about the visible food's cooking condition.

Follow these instructions:

1. Identify the food if it is recognizable.
2. Examine visible color, texture, browning, charring, and other observable cooking indicators.
3. Determine whether the food visually appears cooked, undercooked, possibly overcooked/burnt, or whether its cooking state cannot be determined.
4. Only draw conclusions supported by visible evidence.
5. Do not assume that dark-colored food is necessarily burnt.
6. Do not invent ingredients, temperatures, elapsed cooking times, or cooking methods.
7. If the food is obscured, unclear, or the image does not contain recognizable food, respond that the cooking state cannot be determined.
8. Never claim that food is safe to eat based on an image alone. Internal cooking temperature cannot be reliably established visually.

Your response must use this format:

Food Identified: [Food name or Unknown]

Cooking Status: [APPEARS_COOKED / APPEARS_UNDERCOOKED / POSSIBLY_OVERCOOKED_OR_BURNT / CANNOT_DETERMINE]

Visual Evidence: [One or two sentences identifying the observable evidence supporting the assessment]

Recommendation: [One short, practical recommendation appropriate for the observation]

Keep the response clear, concise, factual, and suitable for an elderly user's cooking assistant.

Do not output hidden reasoning, unnecessary explanations, or a long conversational introduction."""


class ModelLoadError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


def validate_image(path: Path) -> Path:
    """Ensure the path exists, is a file, and decodes as an image."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"File is not a readable image: {path} ({e})") from e
    return path


class CookingVLM:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            self.model, self.processor = load(model_id)
            self.config = load_config(model_id)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{model_id}': {e}") from e

    def assess(
        self,
        image_path: Path,
        prompt: str = COOKING_PROMPT,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
    ):
        """Run the image + prompt through the VLM. Returns the GenerationResult."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        image_path = validate_image(image_path)
        try:
            # Inserts Qwen3-VL's <|vision_start|><|image_pad|><|vision_end|> tokens.
            formatted = apply_chat_template(
                self.processor, self.config, prompt, num_images=1
            )
            result = generate(
                self.model,
                self.processor,
                formatted,
                image=[str(image_path)],
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(f"Inference failed on {image_path}: {e}") from e
        if not result.text.strip():
            raise InferenceError("Model returned an empty response.")
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B cooking-state check")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    bar = "=" * 40
    print(f"{bar}\nELDERLY COOKING ASSISTANT\nQwen3-VL-4B Visual Inference\n{bar}\n")
    print(f"Model: {MODEL_ID}")

    try:
        image_path = validate_image(args.image)
        print(f"Image: {image_path}\n")

        print("Loading model...")
        t0 = time.perf_counter()
        vlm = CookingVLM()
        print(f"Model loaded. ({time.perf_counter() - t0:.1f} seconds)\n")

        print("Running image inference...\n")
        t0 = time.perf_counter()
        result = vlm.assess(image_path, max_tokens=args.max_tokens)
        elapsed = time.perf_counter() - t0
    except (FileNotFoundError, ValueError, ModelLoadError, InferenceError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("========== MODEL RESPONSE ==========\n")
    print(result.text.strip())
    print(f"\n{'=' * 38}")
    print(f"Inference Time: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {result.prompt_tokens} (includes image tokens), "
          f"generated tokens: {result.generation_tokens}")
    print("=" * 38)
    return 0


if __name__ == "__main__":
    sys.exit(main())
