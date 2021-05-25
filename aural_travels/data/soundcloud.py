import os
import json
import math
import logging
import numpy as np
from io import BytesIO
from random import Random

from mutagen.id3 import ID3
from PIL import Image
import librosa
from librosa.feature import mfcc

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scdata


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Calculated by features notebook over 1000 examples sampled from the training data.
MFCC_MEAN = [-129.6565,  93.0169,   6.5002,   22.2244,    4.3236,   10.2308,
             -0.5253,    6.3963,   -2.1541,    4.7143,   -2.3346,    3.7656,
             -2.2059,    2.7291,   -2.3556,    2.6430,   -2.8859,    2.2499,
             -2.0655,    2.2221]
MFCC_STD = [147.8568,  55.7701,  34.7229,  22.2324,  17.8979,  16.1369,  14.1328,
            13.1724,   12.0060,  11.6405,  10.6034,  10.4883,   9.8023,   9.4330,
            9.2720,     9.0574,   8.7890,   8.8789,   8.7943,   8.8655]


def load_tracks(data_dir):
    with open(os.path.join(data_dir, 'scdata.json')) as f:
        return list(json.load(f).values())


def tracks_split(tracks, split):
    return [track for track in tracks if track['scdata_split'] == split]


def load_image(data_dir, track_id):
    audio_path = scdata.get_audio_path(os.path.join(data_dir, 'audio'), track_id)
    tags = ID3(audio_path)
    image = Image.open(BytesIO(tags.getall('APIC')[0].data))

    # Remove alpha channel if present.
    image = image.convert('RGB')

    return image


class GenrePredictionDataset(Dataset):
    def __init__(self, data_dir, split, input_transform):
        self.data_dir = data_dir
        self.split = split
        self.input_transform = input_transform

        all_tracks = load_tracks(data_dir)
        self.tracks = tracks_split(all_tracks, split)

        self.idx_to_genre = dict(enumerate(set(
            scdata.map_genre(track['genre']) for track in all_tracks)))
        self.genre_to_idx = {genre: idx for idx, genre in self.idx_to_genre.items()}

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        image = load_image(self.data_dir, track['id'])
        genre_idx = self.genre_to_idx[scdata.map_genre(track['genre'])]

        return self.input_transform(image), genre_idx

    def num_classes(self):
        return len(self.genre_to_idx)


class CoverGenerationDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 image_labels=None,
                 sample_secs=2.0,
                 sample_rate=22050,
                 n_fft=2048, # ~93ms at 22050Hz
                 hop_length=1024, # ~46ms at 22050Hz
                 normalize_mfcc=False,
                 mfcc_mean=MFCC_MEAN,
                 mfcc_std=MFCC_STD,
                 toy_data=False):
        self.data_dir = data_dir
        self.split = split
        self.image_labels = image_labels
        self.sample_secs = sample_secs
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize_mfcc = normalize_mfcc
        if mfcc_mean:
            self.mfcc_mean = torch.tensor(mfcc_mean)
        if mfcc_std:
            self.mfcc_std_inv = 1.0 / torch.tensor(mfcc_std)
        self.toy_data = toy_data

        self.tracks = tracks_split(load_tracks(data_dir), split)

        if toy_data:
            self.tracks = self.tracks[:10]

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track_id = self.tracks[idx]['id']
        track_secs = self.tracks[idx]['duration'] / 1000.0

        if self.split == 'training':
            # Use torch random seed, which is initialized differently for each data worker and for
            # each epoch.
            offset = torch.rand(1).item() * track_secs
        else:
            # Use fixed song-dependent random seed for evaluating on static validation/test sets.
            offset = Random(x=track_id).random() * track_secs

        audio_path = scdata.get_audio_path(os.path.join(self.data_dir, 'audio'), track_id)
        y_padded = np.zeros(int(self.sample_secs * self.sample_rate))
        try:
            y, _ = librosa.load(audio_path,
                                sr=self.sample_rate,
                                mono=True,
                                offset=offset,
                                duration=self.sample_secs)
            y_padded[:y.shape[0]] = y
        except ValueError:
            logger.warn(f'Empty song (id={track_id}, idx={idx}, duration={track_secs})')

        mel = mfcc(y=y_padded,
                   sr=self.sample_rate,
                   n_fft=self.n_fft,
                   hop_length=self.hop_length,
                   center=False)
        mel = torch.tensor(mel, dtype=torch.float).T

        if self.normalize_mfcc:
            mel = (mel - self.mfcc_mean) * self.mfcc_std_inv

        result = (mel,)
        if self.image_labels is not None:
            result += (self.image_labels[idx],)

        return result

    def num_features(self):
        return 20

    def num_samples(self):
        # FIXME
        return int(self.sample_secs * self.sample_rate / self.hop_length) - 1