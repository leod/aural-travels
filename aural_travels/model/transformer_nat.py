"""
Non-autoregressive transformer for predicting images (in terms of DALL-E tokens) conditioned on 
an input sequence.

Based on
<https://github.com/lucidrains/DALLE-pytorch/blob/093b9ef4618a8381fd68b648bd177b7097550503/dalle_pytorch/transformer.py>,
but without causal masking.
"""

from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, hidden_size, depth, inner):
        super().__init__()

        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, hidden_size).fill_(init_eps)
        self.scale = nn.Parameter(scale)

        self.inner = inner

    def forward(self, x, **kwargs):
        return self.inner(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, hidden_size, inner):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.inner = inner

    def forward(self, x, **kwargs):
        return self.inner(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.0, mult = 2):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(hidden_size, hidden_size * mult),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_size * mult, hidden_size))

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)


class Residual(nn.Module):
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = x + layer(x, **kwargs)
        return x


class Attention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 dropout):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (hidden_size / num_heads) ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(dropout))

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]

        # Project to query, key, and values.
        # qkv: triplet of [batch_size, seq_len, hidden_size]
        qkv = self.qkv(x).chunk(3, dim=-1)

        # Split attention heads across hidden state for query/key/value each.
        # Q: Is there a particular reason to put the head dimension second?
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads) for t in qkv]
        q = q * self.scale

        # Compute all pairs of query-key dot products and take softmax across sequence.
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attention = torch.softmax(scores, dim=-1)

        # Take weighted sum across sequence according to attention and apply output layer.
        output = torch.einsum('b h i j, b h j d -> b h i d', attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.output(output)

        return output


class AxialAttention(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 axis,
                 context_len,
                 image_size,
                 dropout):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (hidden_size / num_heads) ** -0.5

        assert axis in ['row', 'column']
        self.axis = axis
        self.context_len = context_len
        self.image_size = image_size

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.Dropout(dropout))

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]

        # Project to query, key, and values.
        # qkv: triplet of [batch_size, seq_len, hidden_size]
        qkv = self.qkv(x).chunk(3, dim=-1)

        # Split attention heads across hidden state for query/key/value each. Batch and heads are
        # combined into a single dimension.
        q, k, v = [rearrange(t, 'b n (h d) -> (b h) n d', h=self.num_heads) for t in qkv]
        q = q * self.scale

        # Split into context and image sequences.
        image_seq_len = self.image_size ** 2

        (q_context, q_image), \
        (k_context, k_image), \
        (v_context, v_image) = [(t[:, :self.context_len], t[:, self.context_len:]) for t in (q, k, v)]

        # Context-to-context attention.
        scores_context = einsum('b i d, b j d -> b i j', q_context, k_context)
        attention_context = torch.softmax(scores_context, dim=-1)
        out_context = einsum('b i j, b j d -> b i d', attention_context, v_context)

        # Image-attention...

        # Split image x/y axis into separate dimensions
        split_axis_einops = 'b (y x) c -> b y x c' if self.axis == 'row' else 'b (y x) c -> b x y c'
        merge_axis_einops = 'b a n d -> b (a n) d' if self.axis == 'row' else 'b a n d -> b (n a) d'

        q_image, k_image, v_image = [rearrange(t, split_axis_einops, x=self.image_size)
                                     for t in (q_image, k_image, v_image)]

        # Dot products and softmax.
        scores_image_to_image = einsum('b a i d, b a j d -> b a i j', q_image, k_image)
        scores_image_to_context = einsum('b a i d, b j d', 'b a i j', q_image, k_context)
        scores = torch.cat((scores_image_to_context, scores_image_to_image), dim=-1)
        attention = torch.softmax(scores, dim=-1)

        # Take weighted sums.
        attention_image_to_context = attention[..., :self.context_len]
        attention_image_to_image = attention[..., self.context_len:]

        out_image_to_image = einsum('b a i j, b a j d -> b a i d',
                                    attention_image_to_image,
                                    v_image)
        out_image_to_context = einsum('b a i j, b j d -> b a i d',
                                      attention_image_to_context,
                                      v_context)
        out_image = out_image_to_image + out_image_to_context

        # Merge x/y axis to back single dimension.
        out_image = rearrange(out_image, merge_axis_einops)

        # Prepare output, concatenating across sequence axis and unfolding the combined batch axis.
        output = torch.cat((out_context, out_image), dim=1)
        output = rearrange(output, '(b h) n d -> b n (h d)', h=self.num_heads)
        output = self.output(output)
        return output


class TransformerNAT(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 context_len,
                 image_size,
                 num_heads=8,
                 ffnn_mult=2,
                 ffnn_dropout=0.,
                 attention_dropout=0.,
                 axial_attention=False):
        super().__init__()

        if axial_attention:
            axis = lambda depth: 'column' if (depth - 2) % 4 == 0 else 'row'
            attention = lambda depth: AxialAttention(hidden_size=hidden_size,
                                                     num_heads=num_heads,
                                                     axis = axis(depth),
                                                     context_len=context_len,
                                                     image_size=image_size,
                                                     dropout=attention_dropout)
        else:
            attention = lambda _depth: Attention(hidden_size=hidden_size,
                                                 num_heads=num_heads,
                                                 dropout=attention_dropout)

        block = lambda depth, layer: LayerScale(hidden_size, depth, PreNorm(hidden_size, layer))

        layers = [[block(attention(depth)),
                   block(FeedForward(hidden_size, ffnn_mult=ffnn_mult, ffnn_dropout=ffnn_dropout))]
                  for depth in range(num_layers)]

        self.residual = nn.ModuleList(sum(layers, []))

    def forward(self, x, **kwargs):
        return self.residual(x, **kwargs)
