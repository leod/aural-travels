""" Utilities for loading the FMA dataset. """

import os
import ast
import logging

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Based on:
# https://github.com/mdeff/fma/blob/e3d369c49bce73abe467ccffe4dbd3d94c27d7fd/utils.py


def load_genres(data_dir):
    filepath = os.path.join(data_dir, 'fma_metadata', 'genres.csv')
    return pd.read_csv(filepath, index_col=0)


def load_tracks(data_dir):
    filepath = os.path.join(data_dir, 'fma_metadata', 'tracks.csv')
    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    LIT_COLUMNS = [('track', 'tags'),
                   ('album', 'tags'),
                   ('artist', 'tags'),
                   ('track', 'genres'),
                   ('track', 'genres_all')]
    DAT_COLUMNS = [('track', 'date_created'),
                   ('track', 'date_recorded'),
                   ('album', 'date_created'),
                   ('album', 'date_released'),
                   ('artist', 'date_created'),
                   ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
    CAT_COLUMNS = [('track', 'genre_top'),
                   ('track', 'license'),
                   ('album', 'type'),
                   ('album', 'information'),
                   ('artist', 'bio')]
    SUBSETS = ('small', 'medium', 'large')

    for column in LIT_COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)
    for column in DAT_COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])
    for column in CAT_COLUMNS:
        tracks[column] = tracks[column].astype('category')

    tracks['set', 'subset'] = tracks['set', 'subset'].astype(
        pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    return tracks


def load_albums(data_dir):
    filepath = os.path.join(data_dir, 'fma_metadata_albums.csv')
    albums = pd.read_csv(filepath, index_col=0)

    CAT_COLUMNS = ['genre_top']
    SUBSETS = ['small', 'medium', 'large']
    
    for column in CAT_COLUMNS:
        albums[column] = albums[column].astype('category')

    albums['subset'] = albums['subset'].astype(
        pd.CategoricalDtype(categories=SUBSETS, ordered=True))

    return albums


def get_image_path(data_dir, album_id):
    return os.path.join(data_dir, 'fma_album_covers', f'{album_id}.jpg')


class GenrePredictionDataset(Dataset):
    printed_info = set()

    def __init__(self,
                 data_dir,
                 subset,
                 split,
                 input_transform):
        albums = load_albums(data_dir)

        albums = albums[albums['split'] == split]
        albums = albums[albums['subset'] <= subset]
        albums = albums[albums['has_cover'] == True]
        albums = albums[~albums['genre_top'].isnull()]

        self.data_dir = data_dir
        self.input_transform = input_transform
        self.genre_to_idx = {name: idx for idx, name in enumerate(set(albums['genre_top']))}
        self.albums = albums

        # Reduce logging noise a bit...
        key = '|||'.join([data_dir, subset, split])
        if key not in self.printed_info:
            logging.info(f"Loaded album data from data_dir=`{data_dir}'")
            logging.info(f'Found {len(albums)} albums (having cover and genre_top) in split="{split}", '
                        f'subset="{subset}"')
            logging.info(f'Genre distribution: \n{albums["genre_top"].value_counts(normalize=True)}')
            logging.info(f'genre_to_idx={self.genre_to_idx}')
            self.printed_info.add(key)


    def __len__(self):
        return len(self.albums)

    def __getitem__(self, idx):
        row = self.albums.iloc[idx]

        genre_idx = self.genre_to_idx[row.genre_top]
        image = Image.open(get_image_path(self.data_dir, row.name))

        # Remove alpha channel if present
        image = image.convert('RGB')

        return self.input_transform(image), genre_idx
