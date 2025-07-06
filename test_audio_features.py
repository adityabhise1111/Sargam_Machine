import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
filename = "sample.mp3"
y, sr = librosa.load(filename)

# Print basic info
print(f"Loaded {filename}")
print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
print(f"Sample rate: {sr} Hz")

# ===============================
# 1. TEMPO DETECTION
# ===============================
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#print(f"Estimated Tempo: {tempo:.2f} BPM")
if isinstance(tempo, (np.ndarray, list)):
    print(f"Estimated Tempo: {tempo[0]:.2f} BPM")
else:
    print(f"Estimated Tempo: {tempo:.2f} BPM")

# ===============================
# 2. PITCH DETECTION (F0 Curve)
# ===============================
f0, voiced_flag, voiced_probs = librosa.pyin(
    y, 
    fmin=librosa.note_to_hz('C2'),  # 65 Hz
    fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
    sr=sr
)

# Remove unvoiced portions
times = librosa.times_like(f0)
f0_clean = np.where(np.isnan(f0), 0, f0)

# Print first 10 pitch values
print("First 10 pitch frequencies (Hz):")
print(f0[:10])

# ===============================
# 3. PLOT (Optional)
# ===============================
plt.figure(figsize=(14, 5))
plt.plot(times, f0_clean, label='Pitch (Hz)')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Pitch Contour (F0) Over Time")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
