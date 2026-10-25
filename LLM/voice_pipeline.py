"""Voice pathway for the Elderly Cooking Assistant.

    ESP32 mic audio -> speech-to-text (Whisper) -> Qwen3-VL-4B (camera frame + what
    the cook said) -> Kokoro-82M text-to-speech -> WAV for the ESP32 speaker.

Everything runs locally. `VoicePipeline` loads each model once, so the backend can
hold one instance and call `respond()` per request. The output WAV matches what the
ESP32 plays: 16 kHz, 16-bit signed, mono.

Test with a frame from a video:
    python voice_pipeline.py --video omelette.mp4 --at 80
    python voice_pipeline.py --video omelette.mp4 --at 80 --say "I have started cooking the omelet"
    python voice_pipeline.py --image images.jpeg --audio question.wav
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from inference import MODEL_ID, CookingVLM, InferenceError

TTS_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
TTS_VOICE = "af_heart"  # American English, warm female voice
TTS_SPEED = 0.9  # a little slower than default, easier to follow
TTS_SAMPLE_RATE = 24000  # Kokoro's native rate
STT_MODEL = "base.en"
ESP32_SAMPLE_RATE = 16000
MAX_TOKENS = 120
TEMPERATURE = 0.0

SPOKEN_PROMPT = """You are a friendly voice cooking assistant helping an elderly person cook. Your answer will be read aloud through a small speaker.

Look at the attached photo of their stove or pan.

Reply in two or three short, plain spoken sentences:
1. Say what food you see and how it looks right now (color, texture, browning).
2. Give one simple, practical next step.

Rules:
- Only describe what is actually visible. If you cannot see the food clearly, say so.
- Trust the photo over what the cook says. If they mention food you cannot see in the pan, gently tell them you don't see it yet.
- Never say the food is safe to eat; you cannot tell internal temperature from a photo.
- No lists, headings, symbols, or emojis. Just natural speech.{said}"""


def extract_frame(video_path: Path, seconds: float) -> Image.Image:
    """Grab the frame at `seconds` from a video as an RGB PIL image."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 1)
        if not 0 <= seconds < duration:
            raise ValueError(f"--at {seconds}s is outside the video (0-{duration:.1f}s)")
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame at {seconds}s from {video_path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def to_esp32_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Float audio -> 16 kHz, 16-bit, mono WAV bytes (the format the ESP32 plays)."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != ESP32_SAMPLE_RATE:
        g = gcd(ESP32_SAMPLE_RATE, sample_rate)
        audio = resample_poly(audio, ESP32_SAMPLE_RATE // g, sample_rate // g)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1.0, 1.0), ESP32_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def load_stt(model: str = STT_MODEL, workers: int = 1):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device="cpu", compute_type="int8", num_workers=workers)


def load_tts(model_id: str = TTS_MODEL_ID):
    from mlx_audio.tts.utils import load_model

    return load_model(model_id)


def transcribe_wav(stt, wav_bytes: bytes) -> str:
    """Speech in any WAV format (e.g. the ESP32's 16 kHz mono) -> text."""
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly

        audio = resample_poly(audio, 16000, sr).astype(np.float32)
    segments, _ = stt.transcribe(audio, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def qwen_reply(vlm: CookingVLM, image: Image.Image, said: str = "", max_tokens: int = MAX_TOKENS):
    """Frame + what the cook said -> GenerationResult whose .text is a short spoken-style reply."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    said_line = (
        f'\n\nThe cook just said: "{said}". Respond to them, but describe only what the photo shows. '
        "Words are not evidence: if that food is not visible in the pan, say you don't see it yet."
        if said
        else ""
    )
    formatted = apply_chat_template(vlm.processor, vlm.config, SPOKEN_PROMPT.format(said=said_line), num_images=1)
    try:
        result = generate(
            vlm.model,
            vlm.processor,
            formatted,
            image=[image.convert("RGB")],
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            verbose=False,
        )
    except Exception as e:
        raise InferenceError(f"Inference failed: {e}") from e
    result.text = result.text.strip()
    if not result.text:
        raise InferenceError("Model returned an empty response.")
    return result


def kokoro_wav(tts, text: str, voice: str = TTS_VOICE, speed: float = TTS_SPEED) -> bytes:
    """Text -> ESP32-ready WAV bytes via Kokoro-82M."""
    chunks = [
        np.asarray(r.audio, dtype=np.float32)
        for r in tts.generate(text=text, voice=voice, speed=speed, lang_code=voice[0])
    ]
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")
    return to_esp32_wav(np.concatenate(chunks), TTS_SAMPLE_RATE)


class VoicePipeline:
    """All three models in one process. The services in services/ split them up instead."""

    def __init__(self, model_id: str = MODEL_ID):
        self.vlm = CookingVLM(model_id)
        self.tts = load_tts()
        self._stt = None  # loaded on first transcribe(); not needed for text-only tests

    def transcribe(self, wav_bytes: bytes) -> str:
        if self._stt is None:
            self._stt = load_stt()
        return transcribe_wav(self._stt, wav_bytes)

    def think(self, image: Image.Image, said: str = "", max_tokens: int = MAX_TOKENS) -> str:
        return qwen_reply(self.vlm, image, said, max_tokens).text

    def speak(self, text: str) -> bytes:
        return kokoro_wav(self.tts, text)

    def respond(self, image: Image.Image, wav_bytes: bytes | None = None, said: str = ""):
        """Full pathway. Returns (transcript, reply_text, reply_wav_bytes)."""
        if wav_bytes:
            said = self.transcribe(wav_bytes)
        reply = self.think(image, said)
        return said, reply, self.speak(reply)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frame (+ optional speech) -> Qwen -> Kokoro audio reply.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Video to take the frame from")
    src.add_argument("--image", type=Path, help="Image to use as the camera frame")
    parser.add_argument("--at", type=float, default=80.0, help="Seconds into the video (default 80 = 1:20)")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--audio", type=Path, help="WAV of the cook speaking (runs speech-to-text)")
    speech.add_argument("--say", default="", help="What the cook said, as text (skips speech-to-text)")
    parser.add_argument("--out", type=Path, help="Output WAV (default: next to the video/image)")
    args = parser.parse_args()

    source = (args.video or args.image).expanduser().resolve()
    try:
        image = extract_frame(source, args.at) if args.video else Image.open(source).convert("RGB")
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    wav_in = args.audio.expanduser().read_bytes() if args.audio else None
    tag = f"_{int(args.at // 60)}m{int(args.at % 60):02d}s" if args.video else ""
    out = args.out or source.with_name(f"{source.stem}{tag}_response.wav")

    t0 = time.perf_counter()
    pipeline = VoicePipeline()
    t1 = time.perf_counter()
    try:
        said, reply, wav = pipeline.respond(image, wav_in, args.say)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    t2 = time.perf_counter()
    out.write_bytes(wav)

    print(f"Frame:      {source.name}" + (f" @ {args.at:g}s" if args.video else ""))
    if said:
        print(f"Cook said:  {said}")
    print(f"Reply:      {reply}")
    print(f"Audio:      {out}  ({(len(wav) - 44) / 2 / ESP32_SAMPLE_RATE:.1f}s, 16 kHz mono 16-bit)")
    print(f"Load {t1 - t0:.1f}s | respond {t2 - t1:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
