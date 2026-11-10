import sounddevice as sd
from scipy.io.wavfile import write

# Audio settings
duration = 10
sample_rate = 16000

print("Recording started...")
print("Speak something for 10 seconds!")

# Record audio
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)

sd.wait()

# Save recording
write("test_audio.wav", sample_rate, audio)

print("Recording completed!")
print("Audio saved successfully!")