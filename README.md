# aural-travels

Turn Album Covers Into Music If You Want To Do That For Some Reason.

See also this inspiring [Onion Talk](https://www.youtube.com/watch?v=zpNgsU9o4ik).

## Data

We perform experiments on the FMA dataset, which was created by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst and Xavier Bresson ([paper](https://arxiv.org/abs/1612.01840), [GitHub](https://github.com/mdeff/fma)). 
This repository requires the full dataset, `fma_full.zip` (879 GiB), to be downloaded and extracted ahead of time. The dataset contains 106,574 untrimmed tracks from 161 unbalanced genres.

### Discussion
The FMA dataset has a couple of advantages:
1. The music is licensed under Creative Commons.
2. It is standardized, making it easier for others to reproduce my experiments.
3. It covers a wide set of genres (electronic, rock, hip-hop, classical, etc.).

A potential disadvantage of the FMA dataset is that its music might be quite different from commercial data, since it is provided at least to some degree by hobbyists. Furthermore, there is a risk that album covers are not as descriptive as for commercial data: at least for some of the data I looked at, the album covers seemed to be more of an afterthought. However, I hope that we can use this dataset for pre-training, and then try fine-tuning models on commercial data.
