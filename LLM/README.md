# Elderly Cooking Assistant: Local VLM Inference (Phase 1)

Runs local inference with Qwen3-VL-4B-Instruct on one image and prints the visual cooking-state assessment.

## Hardware prerequisites
- Apple Silicon Mac (M1 or later) on macOS 14.0 or newer, with native arm64 Python. MLX does not run on Intel Macs.
- At least 8 GB of unified memory (tested on an M4 with 16 GB) and about 4 GB of free disk for the model cache.

## Model
`mlx-community/Qwen3-VL-4B-Instruct-4bit` is a 4-bit MLX quantization of `Qwen/Qwen3-VL-4B-Instruct` (about 3.1 GB). It downloads into the Hugging Face cache (`~/.cache/huggingface`) on first run. Model weights are not stored in this repo.

## Install
```bash
conda env create -f environment.yml      # creates "cooking-vlm" (Python 3.11)
conda run -n cooking-vlm python -m pip check
```
Verified versions: mlx-vlm 0.7.10, mlx 0.32.4, transformers 5.20.2.

## Run
```bash
conda run -n cooking-vlm python inference.py                       # defaults to ./images.jpeg
conda run -n cooking-vlm python inference.py --image /path/to/pan.jpg
```
Generation is greedy (temperature 0) and capped at 200 tokens (`--max-tokens` to change). The prompt is the `COOKING_PROMPT` constant in `inference.py`. `CookingVLM` loads the model once and exposes `assess(image_path)`, so it can be reused from a FastAPI service later.

## Limitations
- This is visual classification only, not food-safety verification. Internal temperature cannot be seen in an image.
- A 4B model can misidentify foods and confuse dark sauces or seasoning with charring. Lighting, steam, and camera angle affect results.
- The model does not always follow the prompt. In testing it still said an egg was "safe to eat" even though the prompt forbids that claim. Downstream logic must not trust such statements and should use only the parsed `Cooking Status` field.
- Each image is assessed on its own. Trends over time and timers belong to the future decision engine.
