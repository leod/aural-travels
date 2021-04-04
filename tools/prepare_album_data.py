#!/usr/bin/env python3
""" Prepare FMA album-level metadata and download album covers. """

import os
import argparse
import logging
import random
import requests
import shutil
from collections import Counter

import pandas as pd
import numpy as np

from aural_travels.data import fma


def print_album_stats(albums):
    num_albums = len(albums)

    num_have_genre = len(albums[~albums['genre_top'].isnull()])
    num_have_cover = len(albums[~albums['has_cover'].isnull()])
    num_have_genre_cover = len(albums[~albums['genre_top'].isnull() \
        & ~albums['has_cover'].isnull()])

    perc_have_genre = num_have_genre / num_albums * 100.0
    perc_have_cover = num_have_cover / num_albums * 100.0
    perc_have_genre_cover = num_have_genre_cover / num_albums * 100.0

    info = ('Statistics of album data:\n'
            f'    num_albums = {num_albums}\n'
            f'    num_have_genre = {num_have_genre} ({perc_have_genre:.2f}%)\n'
            f'    num_have_cover = {num_have_cover} ({perc_have_cover:.2f}%)\n'
            f'    num_have_genre_cover = {num_have_genre_cover} ({perc_have_genre_cover:.2f}%)')
    logging.info(info)


def print_raw_album_stats(raw_albums):
    num_albums = len(raw_albums)
    num_have_image_url = len(raw_albums[~raw_albums['album_image_file'].isnull()])
    perc_have_image_url = num_have_image_url / num_albums * 100.0

    info = ('Statistics of raw album data:\n'
            f'    num_albums = {num_albums}\n'
            f'    num_have_image_url = {num_have_image_url} ({perc_have_image_url:.2f}%)')
    logging.info(info)


def update_image_file_url(url):
    """ Update FMA's `album_image_file` (from `raw_albums.csv`) to the new URL scheme.

    For details, see this GitHub issue: <https://github.com/mdeff/fma/issues/51>.
    """
    old_prefix = 'https://freemusicarchive.org/file/'
    new_prefix = 'https://freemusicarchive.org/image?file='
    new_suffix = '&width=290&height=290&type=image'
    assert url.startswith(old_prefix)

    return new_prefix + url[len(old_prefix):] + new_suffix


def download_file(url, path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, 'wb') as f:
        shutil.copyfileobj(r.raw, f) 
    

def download_album_covers(data_dir, album_cover_dir, albums):
    raw_album_filepath = os.path.join(data_dir, 'fma_metadata', 'raw_albums.csv')
    
    logging.info(f"Loading FMA raw album metadata from `{raw_album_filepath}'...")
    raw_albums = pd.read_csv(raw_album_filepath, index_col=0)
    print_raw_album_stats(raw_albums)

    logging.info('Proceeding to download album cover images...')

    def f(row):
        path = os.path.join(album_cover_dir, str(row.name) + '.jpg')
        if os.path.exists(path):
            logging.info(f'Already have cover image, skipping: id={row.name}, '
                         f'title="{row.album_title}"')
            return True

        try:
            logging.info(f'Downloading cover image: id={row.name}, title="{row.album_title}"')
            url = update_image_file_url(row.album_image_file)
            download_file(url, path)
            return True
        except Exception as e:
            logging.info(f'Caught exception for: id={row.name}, title="{row.album_title}"')
            logging.info(f'Error: {e}')
            return False

    albums['has_cover'] = raw_albums.apply(f, axis=1)


def prepare_album_data(data_dir, output_file, album_cover_dir):
    logging.info(f"Top-level FMA data directory: `{data_dir}'")
    logging.info(f"Will write album metadata to file: `{output_file}'")
    logging.info(f"Will write album cover image files to directory: `{album_cover_dir}'")

    os.makedirs(album_cover_dir, exist_ok=True)
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

    download_album_covers(data_dir, album_cover_dir, albums)

    logging.info(f"Saving album metadata to `{output_file}'...")
    albums.to_csv(output_file)

    print_album_stats(albums) 

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
