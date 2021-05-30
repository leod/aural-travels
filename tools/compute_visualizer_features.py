"""
Precompute audio features.
"""

import argparse
import os

import librosa
import numpy as np
from librosa.feature import mfcc

def compute_features(path, sample_rate, n_fft, hop_length):
    try:
        y, _ = librosa.load(path,
                            sr=sample_rate,
                            mono=True)
    except ValueError:
        print(f'Empty song (id={track_id}, idx={idx}, duration={track_secs})')
        y = [0.0] * n_fft

    mel = mfcc(y=y,
               sr=sample_rate,
               n_fft=n_fft,
               hop_length=hop_length,
               center=False)
    mel = np.array(mel, dtype=np.float16).T

    suffix = f'_sr{sample_rate}_n_fft{n_fft}_hop_length{hop_length}'
    out_path = os.path.splitext(path)[0] + f'{suffix}.npz'
    np.savez(out_path, mel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--sample_rate', default=22050, type=int)
    parser.add_argument('--n_fft', default=2048, type=int)
    parser.add_argument('--hop_length', default=1024, type=int)
    args = parser.parse_args()

    compute_features(**vars(args))
