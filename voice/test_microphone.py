import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 5

print("Recording starts now. Speak for 5 seconds!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write("test_recording.wav", audio, SAMPLE_RATE)

print("Recording complete!")
print("Saved as test_recording.wav")