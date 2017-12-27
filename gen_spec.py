from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display

n_fft = 1024
lag = 2
n_mels = 138
fmin = 27.5
fmax = 16000.
max_size = 3


source = '.'
EXT = ('.wav')
for root, dirs, filenames in os.walk(source):
    for f in filenames:
        if f.endswith(EXT):
            print (f)
            fullpath = os.path.join(source, f)
            log = open(fullpath, 'r')

            # Read wav
            y, sr = librosa.load(f,
                     sr=44100,
                     duration=10,
                     offset=0)
            hop_length = int(librosa.time_to_samples(1. / 200, sr=sr))
            # Generate Spectrogram array
            S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                   hop_length=hop_length,
                                   fmin=fmin,
                                   fmax=fmax,
                                   n_mels=n_mels)

            np.savetxt(f + ".csv", S, delimiter=",")

            # Plot Spectrogram
            plt.figure(figsize=(20, 4))
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                         y_axis='mel', x_axis='time', sr=sr,
                         hop_length=hop_length, fmin=fmin, fmax=fmax)

            plt.savefig(f + ".png")


