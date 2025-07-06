import librosa
import numpy as np
from mido import Message, MidiFile, MidiTrack, bpm2tempo
from tqdm import tqdm

# Parameters
filename = "Tum Jo Aaye _ Once Upon A Time In Mumbai_ Pritam _ Ajay Devgn, Kangana Ranaut (128 kbps).mp3"
output_midi = "output_cleaned_" + filename + ".mid"
tempo_bpm = 120  # adjust if needed
ticks_per_beat = 480

# Load and extract pitch
print(f"Loading {filename}...")
y, sr = librosa.load(filename)
print(f"Loaded. Duration: {librosa.get_duration(y=y, sr=sr):.2f}s, Sample rate: {sr} Hz")

print("Extracting pitch (f0)...")
f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                   fmax=librosa.note_to_hz('C7'), sr=sr)
times = librosa.times_like(f0)

# Smooth and filter pitch data
print("Converting frequencies to MIDI notes...")
midi_notes = []
for f in tqdm(f0):
    if np.isnan(f):
        midi_notes.append(None)
    else:
        midi_notes.append(int(np.round(librosa.hz_to_midi(f))))

# Group consecutive notes
grouped_notes = []
current_note = None
start_time = None

for i, note in enumerate(midi_notes):
    time = times[i]
    if note is not None and note != current_note:
        if current_note is not None:
            grouped_notes.append((current_note, start_time, time))
        current_note = note
        start_time = time
    elif note != current_note:
        if current_note is not None:
            grouped_notes.append((current_note, start_time, time))
        current_note = None
        start_time = None

# Add last note
if current_note is not None and start_time is not None:
    grouped_notes.append((current_note, start_time, times[-1]))

# Write to MIDI
print("Writing MIDI...")
mid = MidiFile(ticks_per_beat=ticks_per_beat)
track = MidiTrack()
mid.tracks.append(track)

tempo = bpm2tempo(tempo_bpm)
track.append(Message('program_change', program=0, time=0))

def seconds_to_ticks(seconds):
    beats = seconds * tempo_bpm / 60
    return int(beats * ticks_per_beat)

last_tick = 0
for note, start, end in grouped_notes:
    start_tick = seconds_to_ticks(start)
    duration_ticks = seconds_to_ticks(end - start)
    delta = max(0, start_tick - last_tick)

    track.append(Message('note_on', note=note, velocity=64, time=delta))
    track.append(Message('note_off', note=note, velocity=64, time=duration_ticks))
    last_tick = start_tick + duration_ticks

mid.save(output_midi)
print(f"✅ MIDI saved to {output_midi}")
