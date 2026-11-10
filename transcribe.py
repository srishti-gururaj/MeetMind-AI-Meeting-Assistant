from faster_whisper import WhisperModel

print("Loading Whisper model...")

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")

print("Transcribing audio...")

segments, info = model.transcribe(
    "test_audio.wav",
    beam_size=5
)

transcript = ""

for segment in segments:
    transcript += segment.text + " "

print("\nTRANSCRIPTION:\n")
print(transcript)

# Save transcription into a text file
with open("transcript.txt", "w", encoding="utf-8") as file:
    file.write(transcript.strip())

print("\nTranscription saved successfully!")