import torch
from torch import nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from aural_travels.model.transformer_nat import TransformerNAT


class BottleneckGen(nn.Module):
    def __init__(self,
                 image_repr,
                 audio_seq_len,
                 audio_num_features,
                 hidden_size,
                 num_enc_layers,
                 num_dec_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout):
        super().__init__()

        self.image_repr = image_repr
        self.audio_seq_len = audio_seq_len
        self.audio_num_features = audio_num_features
        self.hidden_size = hidden_size
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.ffnn_dropout = ffnn_dropout

        for param in image_repr.parameters():
            param.requires_grad = False

        self.image_seq_len = self.image_repr.grid_size() ** 2

        self.audio_input = nn.Linear(audio_num_features, hidden_size)
        self.audio_pos_emb = nn.Embedding(audio_seq_len, hidden_size)
        self.audio_encoder = TransformerNAT(hidden_size=hidden_size,
                                            num_layers=num_enc_layers)

        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(image_repr.grid_size(),
                                                                   image_repr.grid_size()))
        self.image_decoder = TransformerNAT(hidden_size=hidden_size,
                                            num_layers=num_dec_layers) 
        self.image_output = nn.Sequential(nn.LayerNorm(hidden_size),
                                          nn.Linear(hidden_size, image_repr.vocab_size()))

    def _audio_input(self, audio_seq):
        assert audio_seq.shape[1] == self.audio_seq_len
        assert audio_seq.shape[2] == self.audio_num_features

        audio_emb = self.audio_input(audio_seq)
        audio_emb += self.audio_pos_emb(torch.arange(audio_emb.shape[1], device=audio_emb.device))

        return audio_emb

    def forward(self, audio_seq, target_image_seq):
        logits = self.calc_logits(audio_seq)
        logits = torch.transpose(logits, 1, 2)

        loss = F.cross_entropy(logits, target_image_seq)
        return loss

    def calc_logits(self, audio_seq):
        audio_emb = self._audio_input(audio_seq)
        audio_emb = self.audio_encoder(audio_emb)
        audio_emb = torch.mean(audio_emb, dim=1)[:, None, :]
        audio_emb = torch.tile(audio_emb, (self.image_seq_len, 1))
        audio_emb += self.image_pos_emb(audio_emb)
        image_emb = self.image_decoder(audio_emb)
        return self.image_output(image_emb)

    def generate_image_seq(self,
                           audio_seq,
                           temperature=1.0,
                           top_k=0):
        logits = self.calc_logits(audio_seq)

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits / temperature, dim=-1)

        probs = rearrange(probs, 'b n v -> (b n) v')
        image_seq = torch.multinomial(probs, 1)[..., 0]
        image_seq = rearrange(image_seq, '(b n) -> b n', b=audio_seq.shape[0])

        return image_seq
