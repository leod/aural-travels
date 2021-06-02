"""
Precompute audio features.
"""

import argparse
import os
import sys

import librosa
import audioread
import numpy as np
from librosa.feature import mfcc


def compute_features(path, sample_rate, n_fft, hop_length):
    suffix = f'_sr{sample_rate}_n_fft{n_fft}_hop_length{hop_length}'
    out_path = os.path.splitext(path)[0] + f'{suffix}.npz'
    if os.path.exists(out_path):
        #sys.stderr.write(f'Skipping {path} (already exists)\n')
        #sys.stderr.flush()
        return

    print(path)
    try:
        y, _ = librosa.load(path,
                            sr=sample_rate,
                            mono=True)
    except (EOFError, audioread.NoBackendError):
        sys.stderr.write(f'Empty song (path={path})\n')
        sys.stderr.flush()

        y = np.array([0.0] * n_fft)

    mel = mfcc(y=y,
               sr=sample_rate,
               n_fft=n_fft,
               hop_length=hop_length,
               center=False)
    mel = np.array(mel, dtype=np.float16).T

    np.savez(out_path, mel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--sample_rate', default=22050, type=int)
    parser.add_argument('--n_fft', default=2048, type=int)
    parser.add_argument('--hop_length', default=1024, type=int)
    args = parser.parse_args()

    for path in args.paths:
        try:
            compute_features(path,
                             sample_rate=args.sample_rate,
                             n_fft=args.n_fft,
                             hop_length=args.hop_length)
        except Exception as e:
            sys.stderr.write(f'Exception {type(e)} for path {path}')
            sys.stderr.flush()
