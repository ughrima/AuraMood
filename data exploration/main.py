# main.ipynb

import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# Set the local path to your dataset
dataset_path = '../TESS Toronto emotional speech_set_data/'

# List all .wav files in the dataset folder
files = []
for subdir, dirs, file_list in os.walk(dataset_path):
    for file in file_list:
        if file.endswith('.wav'):
            files.append(os.path.join(subdir, file))

# Display the number of audio files found
print(f"Total audio files found: {len(files)}")

# Visualize a waveform of the first file
audio_sample, sr = librosa.load(files[0])

plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio_sample, sr=sr)
plt.title("Waveform of Audio Sample")
plt.show()

# Visualize the Mel-spectrogram of the first file
S = librosa.feature.melspectrogram(y=audio_sample, sr=sr, n_mels=128)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()

# Optionally, visualize multiple features across the dataset

# Feature extraction for one file
mfccs = np.mean(librosa.feature.mfcc(y=audio_sample, sr=sr, n_mfcc=40).T, axis=0)

print(f"MFCC shape: {mfccs.shape}")
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()

