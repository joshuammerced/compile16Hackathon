"""Timed Qwen3-VL-4B cooking-state inference.

Despite the filename, this uses MLX-VLM (via inference.CookingVLM), not the vLLM engine.
"""

import time

_START = time.perf_counter()  # before heavy imports so total time includes them

import argparse
import sys
from pathlib import Path

from inference import (
    COOKING_PROMPT,
    DEFAULT_IMAGE,
    MAX_TOKENS,
    MODEL_ID,
    CookingVLM,
    InferenceError,
    ModelLoadError,
    validate_image,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Timed Qwen3-VL-4B cooking-state check (MLX-VLM)")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    print(f"Model: {MODEL_ID}")
    try:
        image_path = validate_image(args.image)
        print(f"Image: {image_path}\n")

        print("Loading model...")
        t0 = time.perf_counter()
        vlm = CookingVLM()
        load_time = time.perf_counter() - t0

        print("Running image inference...\n")
        t0 = time.perf_counter()
        result = vlm.assess(image_path, prompt=COOKING_PROMPT, max_tokens=args.max_tokens)
        gen_time = time.perf_counter() - t0
    except (FileNotFoundError, ValueError, ModelLoadError, InferenceError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    total_time = time.perf_counter() - _START

    print("========== MODEL RESPONSE ==========\n")
    print(result.text.strip())
    print("\n============== TIMING ==============")
    print(f"Model load time:       {load_time:.2f} s")
    print(f"Inference time:        {gen_time:.2f} s  (image encoding + prefill + decoding)")
    print(f"Total execution time:  {total_time:.2f} s  (script start to finish, incl. library imports)")
    print(f"Tokens:                {result.prompt_tokens} prompt, {result.generation_tokens} generated")
    if result.generation_tokens:
        print(f"Generated tokens/sec:  {result.generation_tokens / gen_time:.1f}  (generated tokens / inference time)")
        print(f"Decode tokens/sec:     {result.generation_tps:.1f}  (MLX-VLM, decoding only)")
    print("====================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
