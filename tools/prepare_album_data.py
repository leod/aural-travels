#!/usr/bin/env python3
""" Prepare FMA album-level metadata and download album covers. """

import os
import argparse
import logging
import random
from collections import Counter

import pandas as pd
import numpy as np

from aural_travels.data import fma

def print_album_stats(albums):
    num_albums = len(albums)
    num_have_genre = len(albums[~albums['genre_top'].isnull()])
    perc_have_genre = num_have_genre / num_albums * 100.0

    info = ('Statistics of album data:\n'
            f'    num_albums = {num_albums}\n'
            f'    num_have_genre = {num_have_genre} ({perc_have_genre:.2f}%)')
    logging.info(info)

def prepare_album_data(data_dir, output_file, album_cover_dir):
    logging.info(f"Top-level FMA data directory: `{data_dir}'")
    logging.info(f"Will write album metadata to file: `{output_file}'")
    logging.info(f"Will write album cover image files to directory: `{album_cover_dir}'")

    os.makedirs(args.album_cover_dir, exist_ok=True)
    tracks = fma.load_tracks(data_dir)

    albums = []
    for album_id, album_tracks in tracks.groupby([('album', 'id')]):
        album_title = album_tracks['album', 'title'].iloc[0]
        num_tracks = len(album_tracks)

        # For most albums, each track has the same top genre. For the other albums, we take the most
        # frequent genre amongst their tracks.
        genre_top_tracks = list(album_tracks['track', 'genre_top'])
        genre_top = Counter([genre for genre in genre_top_tracks if isinstance(genre, str)]) \
            .most_common(1)
        genre_top = genre_top[0][0] if genre_top else np.nan

        albums.append({
            'id': album_id,
            'title': album_title,
            'genre_top': genre_top,
            'track_ids': list(album_tracks.index),
        })
        
        # Log some random samples (0.1%) for inspection
        if random.random() < 0.001:
            info = ('Sample:\n'
                    f'    album_id={album_id}\n'
                    f'    title={album_title}\n'
                    f'    num_tracks={num_tracks}\n'
                    f'    genre_top_tracks={genre_top_tracks}\n'
                    f'    genre_top={genre_top}')
            logging.info(info)

    albums = pd.DataFrame(albums)
    albums.set_index('id', inplace=True)

    print_album_stats(albums) 

    logging.info(f"Saving album metadata to `{output_file}'...")
    albums.to_csv(output_file)

    logging.info('All done')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',
                        help='Directory to the FMA dataset',
                        required=True)
    parser.add_argument('--output_file',
                        help='File to write album metadata to',
                        required=True)
    parser.add_argument('--album_cover_dir',
                        help='Directory to download album cover image files to',
                        required=True)
    args = parser.parse_args()

    prepare_album_data(**vars(args))
