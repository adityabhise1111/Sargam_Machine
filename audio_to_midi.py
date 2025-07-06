import librosa
import numpy as np
from mido import Message, MidiFile, MidiTrack
from tqdm import tqdm

# Load audio
filename = "Tum Jo Aaye _ Once Upon A Time In Mumbai_ Pritam _ Ajay Devgn, Kangana Ranaut (128 kbps).mp3"
print(f"Loading {filename}...")
y, sr = librosa.load(filename)
print(f"Loaded. Duration: {librosa.get_duration(y=y, sr=sr):.2f}s, Sample rate: {sr} Hz")

# Extract pitch using pYIN
print("Extracting pitch (f0)...")
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, 
    fmin=librosa.note_to_hz('C2'), 
    fmax=librosa.note_to_hz('C7'),
    sr=sr
)

# Convert f0 to MIDI notes (filtering out NaNs)
print("Converting frequencies to MIDI notes...")
midi_notes = []
times = librosa.times_like(f0)

for f, t in tqdm(zip(f0, times), total=len(times)):
    if not np.isnan(f):
        midi = librosa.hz_to_midi(f)
        midi_notes.append((int(np.round(midi)), t))

# Create MIDI file
print("Creating MIDI file...")
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

last_time = 0
for note, t in midi_notes:
    delta_time = int((t - last_time) * 1000)  # Convert to milliseconds
    delta_ticks = int(delta_time * mid.ticks_per_beat / 500)  # Rough conversion
    track.append(Message('note_on', note=note, velocity=64, time=delta_ticks))
    track.append(Message('note_off', note=note, velocity=64, time=delta_ticks))
    last_time = t

# Save
output_file = "output/output.mid"
mid.save(output_file)
print(f"✅ MIDI file saved as {output_file}")
