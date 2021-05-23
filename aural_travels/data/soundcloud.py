import os
import json
import math
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
                 normalize_mfcc=False,
                 mfcc_mean=None,
                 mfcc_std=None):
        self.data_dir = data_dir
        self.split = split
        self.image_labels = image_labels
        self.sample_secs = sample_secs
        self.normalize_mfcc = normalize_mfcc
        if mfcc_mean:
            self.mfcc_mean = torch.tensor(mfcc_mean)
        if mfcc_std:
            self.mfcc_std_inv = 1.0 / torch.tensor(mfcc_std)

        self.tracks = tracks_split(load_tracks(data_dir), split)

        self.sr = 22050
        self.n_fft = 2048 # ~93ms
        self.hop_length = 1024 # ~46ms

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
        y, _ = librosa.load(audio_path,
                            sr=self.sr,
                            mono=True,
                            offset=offset,
                            duration=self.sample_secs)
        mel = mfcc(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, center=False)
        mel = torch.tensor(mel)
        mel = F.pad(mel, (0, self.num_samples() - mel.size()[1])).T

        if self.normalize_mfcc:
            mel = (mel - self.mfcc_mean) * self.mfcc_std_inv

        result = (mel,)
        if self.image_labels:
            result += (self.image_labels[idx],)

        return result

    def num_features(self):
        return 20

    def num_samples(self):
        # FIXME
        return int(self.sample_secs * self.sr / self.hop_length) - 1