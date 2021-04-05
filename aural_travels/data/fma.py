""" Utilities for loading the FMA dataset. """

import os
import ast
import logging

import pandas as pd

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

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


class GenrePredictionDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 subset):
        ...
