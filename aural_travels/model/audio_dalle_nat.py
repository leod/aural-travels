import torch
from torch import nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange

from aural_travels.model.transformer_nat import TransformerNAT


class AudioDALLENAT(nn.Module):
    def __init__(self,
                 image_repr,
                 audio_seq_len,
                 audio_num_features,
                 hidden_size, 
                 num_layers,
                 num_heads,
                 attention_dropout,
                 ffnn_dropout,
                 axial_attention):
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

        self.image_seq_len = self.image_repr.grid_size() ** 2

        self.image_emb = nn.Embedding(self.image_repr.vocab_size() + 1, hidden_size) # +1 for <bos>
        self.audio_pos_emb = nn.Embedding(audio_seq_len + 1, hidden_size) # +1 for <bos>

        # FIXME: Does the order of VQGAN's tokens work with axial positional embedding?
        self.image_pos_emb = AxialPositionalEmbedding(hidden_size,
                                                      axial_shape=(image_repr.grid_size(),
                                                                   image_repr.grid_size()))

        self.transformer = TransformerNAT(hidden_size=hidden_size,
                                          num_layers=num_layers,
                                          context_len=self.audio_seq_len + 1,
                                          grid_size=self.image_repr.grid_size(),
                                          axial_attention=axial_attention)
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

    def forward(self, audio_seq, target_image_seq, corrupt_image_seq, return_logits=False):
        audio_emb = self._audio_input(audio_seq)
        corrupt_image_emb = self._image_input(corrupt_image_seq)

        input = torch.cat((audio_emb, corrupt_image_emb), dim=1)

        output = self.transformer(input)
        output = output[:, audio_emb.shape[1]+1:, :]

        logits = self.output(output)
        logits_for_ce = torch.transpose(logits, 1, 2)

        loss = F.cross_entropy(logits_for_ce, target_image_seq)

        if return_logits:
            return loss, logits
        else:
            return loss

    @torch.no_grad()
    def generate_image_seq(self,
                           audio_seq,
                           corrupt_image_seq=None,
                           temperature=1.0,
                           top_k=0,
                           return_logits=False,
                           map_logits=None):
        audio_emb = self._audio_input(audio_seq) 

        if corrupt_image_seq is None:
            corrupt_image_seq = torch.randint(self.image_repr.vocab_size(),
                                              (audio_seq.shape[0], self.image_seq_len),
                                              device=audio_emb.device)

        corrupt_image_emb = self._image_input(corrupt_image_seq)
        input = torch.cat((audio_emb, corrupt_image_emb), dim=1)
        output = self.transformer(input)
        output = output[:,self.audio_seq_len+1:, :]

        logits = self.output(output)
        if map_logits:
            logits = map_logits(logits)
        new_logits = torch.clone(logits)

        if top_k > 0:
            indices_to_remove = new_logits < torch.topk(new_logits, top_k)[0][..., -1, None]
            new_logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(new_logits / temperature, dim=-1)

        probs = rearrange(probs, 'b n v -> (b n) v')
        image_seq = torch.multinomial(probs, 1)[..., 0]
        image_seq = rearrange(image_seq, '(b n) -> b n', b=audio_seq.shape[0])

        if return_logits:
            return image_seq, logits
        else:
            return image_seq
