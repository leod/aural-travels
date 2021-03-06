import torch
from torch import nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from dalle_pytorch.transformer import Transformer


class AudioDALLE(nn.Module):
    # Closely following this:
    # <https://github.com/lucidrains/DALLE-pytorch/blob/bdb04280c9ab55eb20f86b375dc1aad20fbd5315/dalle_pytorch/dalle_pytorch.py#L308>,
    # but with continuous audio features rather than discrete tokens as input.

    def __init__(self,
                 image_repr,
                 audio_seq_len,
                 audio_num_features,
                 hidden_size, 
                 num_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout):
        super().__init__()

        self.image_repr = image_repr
        self.audio_seq_len = audio_seq_len
        self.audio_num_features = audio_num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffnn_dropout = ffnn_dropout

        for param in image_repr.parameters():
            param.requires_grad = False

        self.image_seq_len = image_repr.grid_size() ** 2
        self.total_seq_len = audio_seq_len + self.image_seq_len

        # Sparse attention order as in the DALL-E paper.
        attention_types = ['axial_col' if (i - 2) % 4 == 0 else 'axial_row'
                           for i in range(num_layers - 1)]
        attention_types += ['conv_like']

        self.image_emb = nn.Embedding(image_repr.vocab_size() + 1, hidden_size) # +1 for <bos>
        self.audio_pos_emb = nn.Embedding(audio_seq_len + 1, hidden_size) # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(image_repr.grid_size(),
                                                                   image_repr.grid_size()))

        self.transformer = Transformer(dim=hidden_size,
                                       causal=True,
                                       seq_len=self.total_seq_len,
                                       depth=num_layers,
                                       heads=num_heads,
                                       dim_head=hidden_size // num_heads,
                                       reversible=False,
                                       attn_dropout=attention_dropout,
                                       ff_dropout=ffnn_dropout,
                                       attn_types=attention_types,
                                       image_fmap_size=image_repr.grid_size(),
                                       sparse_attn=True)
        self.input = nn.Linear(audio_num_features, hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size),
                                    nn.Linear(hidden_size, image_repr.vocab_size()))

    def _audio_input(self, audio_seq):
        assert audio_seq.shape[1] == self.audio_seq_len
        assert audio_seq.shape[2] == self.audio_num_features

        audio_emb = self.input(audio_seq)
        audio_emb += self.audio_pos_emb(torch.arange(audio_emb.shape[1], device=audio_emb.device))

        return audio_emb

    def _image_input(self, image_seq):
        # We place a <bos> token between audio and image to kick off image token prediction.
        image_seq_with_bos = F.pad(image_seq, (1, 0), value=self.image_repr.vocab_size())
        image_emb = self.image_emb(image_seq_with_bos)
        image_emb[:, 1:, :] += self.image_pos_emb(image_emb[:, 1:, :])
        image_emb[:, 0, :] += self.audio_pos_emb(torch.tensor(self.audio_seq_len,
                                                              device=image_seq.device,
                                                              dtype=torch.long))
        return image_emb

    def forward(self, audio_seq, image_seq):
        audio_emb = self._audio_input(audio_seq)
        image_emb = self._image_input(image_seq)

        input = torch.cat((audio_emb, image_emb[:, :-1]), dim=1)

        output = self.transformer(input)
        output = output[:, audio_emb.shape[1]:, :]

        logits = self.output(output)
        logits = torch.transpose(logits, 1, 2)

        loss = F.cross_entropy(logits, image_seq)

        return loss

    @torch.no_grad()
    def generate_images(self,
                        audio_seq,
                        temperature=1.0,
                        top_k=0,
                        map_logits=None):
        audio_emb = self._audio_input(audio_seq) 

        image_seq = torch.zeros((audio_seq.size()[0], 0), dtype=torch.long, device=audio_seq.device)

        for step in range(self.image_seq_len):
            image_emb = self._image_input(image_seq)
            input = torch.cat((audio_emb, image_emb), dim=1)

            output = self.transformer(input)
            logits = self.output(output)

            if map_logits:
                new_logits = map_logits(logits)

            new_logits = torch.clone(logits[:, -1, :])

            if top_k > 0:
                indices_to_remove = new_logits < torch.topk(new_logits, top_k)[0][..., -1, None]
                new_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(new_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            image_seq = torch.cat((image_seq, sample), dim=-1)

        return image_seq, logits #self.vae.decode(image)
