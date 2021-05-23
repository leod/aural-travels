# FMA Dataset

## Preparation
The following files of the [FMA dataset](https://github.com/mdeff/fma) need to be downloaded and
extracted ahead of time:
- [`fma_full.zip`](https://os.unil.cloud.switch.ch/fma/fma_full.zip) (879 GiB)
- [`fma_metadata.zip`](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) (342 MiB)

Extract them into a directory, such that you have a `fma_full` directory and a `fma_metadata`
directory next to each other. This data directory needs to be specified at several points in this
repository.

Once the data has been downloaded and extracted, we need to enrich it with album-level information.
The reason is that the FMA metadata is stored on track-level, while in this work we are more
interested in album-level classification and prediction. Also, we need to download album covers,
which are [not included](https://github.com/mdeff/fma/issues/51) in the FMA dataset.

Let `$DATA_DIR` be your FMA data directory, then run the following command to add album-level
information:
```
PYTHONPATH=. tools/fma_prepare_album_data.py \
    --data_dir $DATA_DIR \
    --output_file $DATA_DIR/fma_metadata_albums.csv \
    --album_cover_dir $DATA_DIR/fma_album_covers
```

Most albums have a cover image, but only about half of them have genre metadata:
```
num_albums = 14854
num_have_genre = 7273 (48.96%)
num_have_cover = 13509 (90.95%)
num_have_genre_cover = 6531 (43.97%)
```

## Discussion
The FMA dataset has a couple of advantages:
1. The music is licensed under Creative Commons.
2. It is standardized, making it easier for others to reproduce these experiments.
3. It covers a wide set of genres (electronic, rock, hip-hop, classical, etc.).

A potential disadvantage of the FMA dataset is that its music might be quite different from
commercial data, since it is provided at least to some degree by hobbyists. Furthermore, there is a
risk that album covers are not as descriptive as for commercial data: at least for some of the data
we looked at, the album covers seemed to be more of an afterthought. However, we hope that we can
use this dataset for pre-training, and then try fine-tuning models on commercial data.
