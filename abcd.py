import librosa
import numpy as np
from tqdm import tqdm

# Step 1: Load audio
filename = "sample.mp3"
print(f"Loading {filename}...")
y, sr = librosa.load(filename)
print(f"Loaded. Duration: {librosa.get_duration(y=y, sr=sr):.2f}s, Sample rate: {sr} Hz")

# Step 2: Extract pitch
print("Extracting pitch (f0)...")
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr
)

# Step 3: Map frequencies to note names
print("Mapping frequencies to note names...")
notes = []
times = librosa.times_like(f0, sr=sr)

for hz in tqdm(f0):
    if np.isnan(hz):
        notes.append(None)
    else:
        note = librosa.hz_to_note(hz)
        notes.append(note)

# Step 4: Print all detected notes
print("\nAll detected notes:")
for t, n in zip(times, notes):
    if n is not None:
        print(f"{t:.2f}s: {n}")

# Step 5: Save results to file
print("\nSaving to output_notes.txt...")
with open("output_notes.txt", "w", encoding="utf-8") as f:
    for t, n in zip(times, notes):
        if n is not None:
            f.write(f"{t:.2f}s: {n}\n")

print("Done!")
