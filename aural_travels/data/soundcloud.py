import os
import json

from io import BytesIO
from mutagen.id3 import ID3
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import scdata


class GenrePredictionDataset(Dataset):
    def __init__(self, data_dir, split, input_transform):
        self.data_dir = data_dir
        self.split = split
        self.input_transform = input_transform

        with open(os.path.join(data_dir, 'scdata.json')) as f:
            self.tracks = list(json.load(f).values())
            self.idx_to_genre = dict(enumerate(set(
                scdata.map_genre(track['genre']) for track in self.tracks)))
            self.genre_to_idx = {genre: idx for idx, genre in self.idx_to_genre.items()}

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        genre_idx = self.genre_to_idx[scdata.map_genre(track['genre'])]
        audio_path = scdata.get_audio_path(os.path.join(self.data_dir, 'audio'), track['id'])
        tags = ID3(audio_path)
        image = Image.open(BytesIO(tags.getall('APIC')[0].data))

        # Remove alpha channel if present
        image = image.convert('RGB')

        return self.input_transform(image), genre_idx

    def num_classes(self):
        return len(self.genre_to_idx)