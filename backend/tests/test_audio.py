import io
import wave

import numpy as np

from app.audio.tts import AudioClipStore, BeepTTS, Speaker, TTSProvider, beep
from app.audio.wav import decode_audio, encode_wav


def assert_16k_mono_16bit(wav: bytes) -> wave.Wave_read:
    w = wave.open(io.BytesIO(wav))
    assert w.getframerate() == 16000
    assert w.getnchannels() == 1
    assert w.getsampwidth() == 2
    assert w.getcomptype() == "NONE"
    return w


def test_beep_format():
    for kind in ("info", "done", "error"):
        w = assert_16k_mono_16bit(beep(kind))
        assert w.getnframes() > 1000


def test_encode_resamples_and_downmixes():
    sr = 24000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    w = assert_16k_mono_16bit(encode_wav(stereo, sr))
    assert abs(w.getnframes() - 16000) <= 1


def test_decode_raw_pcm_and_wav_roundtrip():
    pcm = (np.sin(np.linspace(0, 100, 16000)) * 10000).astype("<i2").tobytes()
    raw = decode_audio(pcm)
    assert raw.shape == (16000,)
    wav = encode_wav(raw)
    back = decode_audio(wav)
    assert np.allclose(raw, back, atol=1e-3)


class BrokenTTS(TTSProvider):
    name = "broken"

    async def synthesize(self, text):
        raise RuntimeError("no network")


async def test_speaker_falls_back_to_beep():
    speaker = Speaker(BrokenTTS())
    clip = speaker.store.get(await speaker.speak("hello"))
    assert_16k_mono_16bit(clip)


async def test_beep_speaker_done_chime():
    speaker = Speaker(BeepTTS())
    clip = speaker.store.get(await speaker.speak("done", chime="done"))
    assert_16k_mono_16bit(clip)


def test_clip_store_keeps_last_n():
    store = AudioClipStore(max_clips=3)
    ids = [store.put(b"x%d" % i) for i in range(5)]
    assert store.get(ids[0]) is None and store.get(ids[1]) is None
    assert store.get(ids[4]) == b"x4"
