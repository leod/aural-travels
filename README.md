# aural-travels

Experiments with audio data.

See also this inspiring [Onion Talk](https://www.youtube.com/watch?v=zpNgsU9o4ik).

> I ask myself: What are two separate things? Can I turn one of those things into the other thing?
> And I did.

## Environment
We've used [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) for these experiments.
Create a new environment from the [`environment.yml`](environment.yml):
```
conda env create -f environment.yml
conda activate aural-travels
```

## Data
We consider two datasets for these experiments:

1. The [`scdata`](https://github.com/leod/scdata) dataset, which contains 35,035 tracks from
   [SoundCloud](https://soundcloud.com/).

2. The FMA dataset, which was created by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst
   and Xavier Bresson ([paper](https://arxiv.org/abs/1612.01840),
   [GitHub](https://github.com/mdeff/fma)). The dataset contains 106,574 untrimmed tracks in 161
   unbalanced genres from the [Free Music Archive](https://freemusicarchive.org/).

   See [`docs/FMA.md`](docs/FMA.md) more information about the dataset, as well as the steps
   necessary to prepare it for the experiments here.

## Experiments
### Predict Genre from Album Cover

### Visualization

### Generate Music from Genre

### Generate Music from Album Cover
