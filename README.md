# aural-travels

Turn Album Covers Into Music If You Want To Do That For Some Reason.

See also this inspiring [Onion Talk](https://www.youtube.com/watch?v=zpNgsU9o4ik).

## Environment
I've used [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) for these experiments, but any Python 3.8.5+ environment should do, hopefully. 

```
conda install --file requirements.txt
```

## Data
We perform experiments on the FMA dataset, which was created by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson ([paper](https://arxiv.org/abs/1612.01840), [GitHub](https://github.com/mdeff/fma)). The dataset contains 106,574 untrimmed tracks in 161 unbalanced genres from the [Free Music Archive](https://freemusicarchive.org/).


### Preparation
This repository requires the full dataset, `fma_full.zip` (879 GiB), to be downloaded and extracted ahead of time. 

### Discussion
The FMA dataset has a couple of advantages:
1. The music is licensed under Creative Commons.
2. It is standardized, making it easier for others to reproduce my experiments.
3. It covers a wide set of genres (electronic, rock, hip-hop, classical, etc.).

A potential disadvantage of the FMA dataset is that its music might be quite different from commercial data, since it is provided at least to some degree by hobbyists. Furthermore, there is a risk that album covers are not as descriptive as for commercial data: at least for some of the data I looked at, the album covers seemed to be more of an afterthought. However, I hope that we can use this dataset for pre-training, and then try fine-tuning models on commercial data.

## Experiments
### Predict Genre from Album Cover

### Generate Music from Genre

### Generate Music from Album Cover
